create or replace function get_distinct_medicine_names()
returns table(medicine_name text) as $$
begin
  return query
    select distinct metadata->>'medicine_name' as medicine_name
    from documents
    where metadata->>'medicine_name' is not null;
end;
$$ language plpgsql;