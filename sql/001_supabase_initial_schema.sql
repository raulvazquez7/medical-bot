    -- 1. Creamos la tabla para guardar nuestros documentos
    create table documents (
      id bigserial primary key,
      content text, -- El texto del chunk
      metadata jsonb, -- Los metadatos como el path jerárquico
      embedding vector(1536) -- El embedding vectorial. 1536 es la dimensión de los embeddings de OpenAI
    );

    -- 2. Creamos una función para buscar documentos por similitud
    create function match_documents (
      query_embedding vector(1536),
      match_count int,
      filter jsonb DEFAULT '{}'
    ) returns table (
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
      where documents.metadata @> filter
      order by documents.embedding <=> query_embedding
      limit match_count;
    end;
    $$;