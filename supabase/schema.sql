-- Supabase schema -- Auth only. Run once in the Supabase SQL editor.
--
-- Safe to re-run: every statement is IF NOT EXISTS / OR REPLACE.
--
-- This file used to mirror the INTRO.txt §2 data model into Postgres as a
-- serving layer. Nothing ever read it: the dashboard reads the committed Parquet
-- straight from its Streamlit Cloud checkout, so the eight data tables and `logs`
-- have been dropped along with the mirror that wrote them. The §2 data model is
-- still enforced -- in src/tobacco/store/parquet_io.py DATASETS and each source
-- module's COLUMNS, which is where it always actually applied.
--
-- What is left is what Supabase is still used for: authenticating the dashboard
-- and looking up the signed-in user's role.
--
-- NOTE ON `users`: INTRO.txt §2 specifies `users(username, password_hash, role)`.
-- This implementation does NOT store password hashes. Supabase Auth owns
-- credentials; keeping a second copy of them in a table the app can read would be
-- a liability with no benefit. The table below holds role assignments only and
-- keys off auth.users.

create table if not exists users (
    id       uuid primary key references auth.users (id) on delete cascade,
    username text,
    -- 'commercial_director' | 'supply_chain_manager' | 'admin'
    role     text not null default 'commercial_director',
    created_at timestamptz default now()
);

-- ===========================================================================
-- row level security
--
-- RLS here gates `users` -- the one table the app reads -- so that a session
-- cannot enumerate other people's role assignments. It is not a confidentiality
-- boundary for the project's data: sales are synthetic and everything else is
-- public macro data, and all of it is committed to a public repository, readable
-- by anyone with no credential at all. Login provides role-based view routing
-- (INTRO.txt §6), not secrecy.
-- ===========================================================================

alter table users enable row level security;

-- A user may read only their own role row.
drop policy if exists users_read_self on users;
create policy users_read_self on users
    for select to authenticated
    using (id = auth.uid());

-- ===========================================================================
-- after signing up your first user in the Supabase Auth UI, assign a role:
--
--   insert into users (id, username, role)
--   values ('<uuid from auth.users>', 'you@example.com', 'admin')
--   on conflict (id) do update set role = excluded.role;
-- ===========================================================================
