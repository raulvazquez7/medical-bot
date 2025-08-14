    -- 1. Creamos la tabla para guardar nuestros documentos
    create table documents (
      id bigserial primary key,
      content text, -- El texto del chunk
      metadata jsonb, -- Los metadatos como el path jerárquico
      embedding vector(1536) -- El embedding vectorial. 1536 es la dimensión de los embeddings de OpenAI
    );

    -- 2. Creamos una función para buscar documentos por similitud
    create or replace function match_documents (
      query_embedding vector(1536),
      match_count int,
      filter_medicines text[] default '{}'
    )
    returns table (
      id bigint,
      content text,
      metadata jsonb,
      similarity float
    )
    language plpgsql
    as $$
    begin
      return query
      select
        documents.id,
        documents.content,
        documents.metadata,
        1 - (documents.embedding <=> query_embedding) as similarity
      from documents
      where 
        (array_length(filter_medicines, 1) is null or documents.metadata->>'medicine_name' = any(filter_medicines))
      order by documents.embedding <=> query_embedding
      limit match_count;
    end;
    $$;

-- Crear un índice HNSW en la columna 'embedding'
CREATE INDEX ON documents
USING hnsw (embedding vector_cosine_ops);