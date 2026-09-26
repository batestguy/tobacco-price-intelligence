-- Supabase schema -- Auth only. Run in the Supabase SQL editor.
--
-- Safe to re-run: every statement is IF NOT EXISTS / OR REPLACE / DROP ... IF
-- EXISTS, and existing rows -- including the first administrator's -- survive.
--
-- This file used to mirror the INTRO.txt §2 data model into Postgres as a
-- serving layer. Nothing ever read it: the dashboard reads the committed Parquet
-- straight from its Streamlit Cloud checkout, so the eight data tables and `logs`
-- have been dropped along with the mirror that wrote them. The §2 data model is
-- still enforced -- in src/tobacco/store/parquet_io.py DATASETS and each source
-- module's COLUMNS, which is where it always actually applied.
--
-- What is left is what Supabase is still used for: authenticating the dashboard,
-- looking up the signed-in user's role, and letting an administrator assign roles
-- from the dashboard's Administration view.
--
-- NOTE ON `users`: INTRO.txt §2 specifies `users(username, password_hash, role)`.
-- This implementation does NOT store password hashes. Supabase Auth owns
-- credentials; keeping a second copy of them in a table the app can read would be
-- a liability with no benefit. The table below holds role assignments only and
-- keys off auth.users.

create table if not exists users (
    id       uuid primary key references auth.users (id) on delete cascade,
    username text,
    role     text,
    created_at timestamptz default now()
);

-- The address the Administration view lists users by.
alter table users add column if not exists email text;

update users u
   set email = a.email
  from auth.users a
 where a.id = u.id
   and u.email is distinct from a.email;

-- `role` is null until an administrator assigns one: null means PENDING, NO
-- ACCESS. There is deliberately no default. Supabase allows public sign-up by
-- default, so a default role would hand a view to anyone who registers.
alter table users alter column role drop not null;
alter table users alter column role drop default;

alter table users drop constraint if exists users_role_check;
alter table users add constraint users_role_check
    check (role is null or role in ('commercial_director', 'supply_chain_manager', 'admin'));

-- ===========================================================================
-- new sign-ups arrive pending
--
-- Every new auth user gets a `users` row with a null role, so an administrator
-- can see them and grant access. The trigger never grants a role.
-- ===========================================================================

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
    insert into public.users (id, email, role)
    values (new.id, new.email, null)
    on conflict (id) do nothing;
    return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
    after insert on auth.users
    for each row execute function public.handle_new_user();

-- Users who signed up before the trigger existed get a pending row too.
insert into users (id, email, role)
select a.id, a.email, null
  from auth.users a
on conflict (id) do nothing;

-- ===========================================================================
-- row level security
--
-- RLS here gates `users` -- the one table the app reads -- so that a session
-- cannot enumerate other people's role assignments, and only an administrator
-- can change one. It is not a confidentiality boundary for the project's data:
-- sales are synthetic and everything else is public macro data, and all of it is
-- committed to a public repository, readable by anyone with no credential at
-- all. Login provides role-based view routing (INTRO.txt §6), not secrecy.
-- ===========================================================================

alter table users enable row level security;

-- Whether the caller is an administrator. SECURITY DEFINER so the lookup reads
-- `users` without going back through the policies below, which would recurse.
create or replace function public.is_admin()
returns boolean
language sql
stable
security definer
set search_path = public
as $$
    select exists (
        select 1 from public.users where id = auth.uid() and role = 'admin'
    );
$$;

-- Callable by signed-in sessions only. For anon, auth.uid() is null so it could
-- only ever answer false, but there is no reason to expose it at all.
revoke execute on function public.is_admin() from public, anon;
grant execute on function public.is_admin() to authenticated;

-- A user may read their own role row.
drop policy if exists users_read_self on users;
create policy users_read_self on users
    for select to authenticated
    using (id = auth.uid());

-- An administrator may read every row...
drop policy if exists users_admin_read_all on users;
create policy users_admin_read_all on users
    for select to authenticated
    using (public.is_admin());

-- ...and change any role but their own. The self-exclusion is enforced here, not
-- only in the dashboard, so an administrator cannot lock themselves out even by
-- calling the REST API directly.
drop policy if exists users_admin_update on users;
create policy users_admin_update on users
    for update to authenticated
    using (public.is_admin() and id <> auth.uid())
    with check (public.is_admin() and id <> auth.uid());

-- Policies choose rows, not columns. Supabase grants `authenticated` every
-- privilege on a new public table, so narrow UPDATE to the one column the
-- dashboard edits: email, username and id stay as the trigger wrote them.
revoke insert, update, delete on public.users from anon, authenticated;
grant update (role) on public.users to authenticated;

-- There is NO insert and NO delete privilege or policy, so no client can do
-- either. Rows are created by the trigger above. Revoking access means setting
-- the role back to null, which keeps the row so the role can be granted again.

-- ===========================================================================
-- granting a role
--
-- New sign-ups arrive with no access. Grant a
-- role by email, here or from the dashboard's Administration view once one
-- administrator exists:
--
--   update users set role = 'admin' where email = 'you@example.com';
--   update users set role = 'supply_chain_manager' where email = 'someone@example.com';
--
-- Roles: 'commercial_director' | 'supply_chain_manager' | 'admin'. The public
-- demo needs no account: it opens a non-admin view without touching Supabase.
--
-- Optional: disable public sign-ups under Authentication -> Sign In / Providers
-- in the Supabase dashboard. It is not required, because a new sign-up is
-- pending and sees nothing until someone grants it a role.
-- ===========================================================================
