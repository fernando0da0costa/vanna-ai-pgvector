import json
import pandas as pd
import psycopg2
import os
from langchain_openai import OpenAIEmbeddings
from vanna.base import VannaBase
from vanna.openai import OpenAI_Chat
import uuid
import hashlib

os.environ["NLS_LANG"] = "AMERICAN_AMERICA.AL32UTF8"

def deterministic_uuid(data: str) -> str:
    """Generate a deterministic UUID based on input data."""
    return str(uuid.UUID(hashlib.md5(data.encode('utf-8')).hexdigest()))

class PGvectorCustomVectorDB(VannaBase):
    def __init__(self, config=None, name_vector='vanna_ia'):
        try:
            self.conn = psycopg2.connect(
                dbname=os.environ.get('DB_NAME'),
                user=os.environ.get('DB_USER'),
                password=os.environ.get('DB_PASSWORD'),
                host=os.environ.get('DB_HOST')
            )
            self.name_vector = name_vector.strip().lower().replace('-', '_')
            self.cursor = self.conn.cursor()
            self.n_results_sql = config.get("n_results_sql", config.get("n_results", 10)) if config else 10
            self.n_results_ddl = config.get("n_results_ddl", config.get("n_results", 10)) if config else 10
            self.n_results_documentation = config.get("n_results_documentation", config.get("n_results", 10)) if config else 10
            self._ensure_tables_exist()
            self._create_indexes()
            self.embed = lambda text: self.get_embedding(text)
        except psycopg2.Error as e:
            raise Exception(f"Failed to connect to database: {e}")

    def to_pgvector_str(self, embedding: list[float]) -> str:
        return "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"

    def generate_embedding(self, text: str) -> list:
        try:
            embedder = OpenAIEmbeddings(api_key=os.environ.get('OPENAI_API_KEY'))
            return embedder.embed_query(text)
        except Exception as e:
            raise Exception(f"Failed to generate embedding: {e}")

    def get_embedding(self, text: str) -> list:
        try:
            return self.generate_embedding(text)
        except Exception as e:
            raise Exception(f"Failed to generate embedding: {e}")

    def add_ddl(self, ddl: str, **kwargs) -> str:
        try:
            id = deterministic_uuid(ddl) + "-ddl"
            embedding = self.embed(ddl)
            embedding_str = self.to_pgvector_str(embedding)
            self.cursor.execute(
                f"INSERT INTO ddl_store_{self.name_vector} (id, question, content, training_data_type, embedding) VALUES (%s, %s, %s, %s, %s::vector)",
                (id, None, ddl, 'ddl', embedding_str)
            )
            self.conn.commit()
            return id
        except psycopg2.Error as e:
            self.conn.rollback()
            raise Exception(f"Failed to add DDL: {e}")

    def add_documentation(self, documentation: str, **kwargs) -> str:
        try:
            id = deterministic_uuid(documentation) + "-doc"
            embedding = self.embed(documentation)
            embedding_str = self.to_pgvector_str(embedding)
            self.cursor.execute(
                f"INSERT INTO documentation_store_{self.name_vector} (id, question, content, training_data_type, embedding) VALUES (%s, %s, %s, %s, %s::vector)",
                (id, None, documentation, 'documentation', embedding_str)
            )
            self.conn.commit()
            return id
        except psycopg2.Error as e:
            self.conn.rollback()
            raise Exception(f"Failed to add documentation: {e}")

    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        try:
            question_sql_json = json.dumps({"question": question, "sql": sql}, ensure_ascii=False)
            id = deterministic_uuid(question_sql_json) + "-sql"
            embedding = self.embed(question_sql_json)
            embedding_str = self.to_pgvector_str(embedding)
            self.cursor.execute(
                f"INSERT INTO question_sql_store_{self.name_vector} (id, question, content, training_data_type, embedding) VALUES (%s, %s, %s, %s, %s::vector)",
                (id, question, sql, 'sql', embedding_str)
            )
            self.conn.commit()
            return id
        except psycopg2.Error as e:
            self.conn.rollback()
            raise Exception(f"Failed to add question-SQL pair: {e}")

    def get_related_ddl(self, question: str, **kwargs) -> list:
        try:
            embedding = self.embed(question)
            embedding_str = self.to_pgvector_str(embedding)
            self.cursor.execute(
                f"SELECT content FROM ddl_store_{self.name_vector} ORDER BY embedding <=> %s LIMIT %s",
                (embedding_str, self.n_results_ddl)
            )
            return [row[0] for row in self.cursor.fetchall()]
        except psycopg2.Error as e:
            raise Exception(f"Failed to retrieve related DDL: {e}")

    def get_related_documentation(self, question: str, **kwargs) -> list:
        try:
            embedding = self.embed(question)
            embedding_str = self.to_pgvector_str(embedding)
            self.cursor.execute(
                f"SELECT content FROM documentation_store_{self.name_vector} ORDER BY embedding <=> %s LIMIT %s",
                (embedding_str, self.n_results_documentation)
            )
            return [row[0] for row in self.cursor.fetchall()]
        except psycopg2.Error as e:
            raise Exception(f"Failed to retrieve related documentation: {e}")

    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        try:
            embedding = self.embed(question)
            embedding_str = self.to_pgvector_str(embedding)
            self.cursor.execute(
                f"SELECT content FROM question_sql_store_{self.name_vector} ORDER BY embedding <=> %s LIMIT %s",
                (embedding_str, self.n_results_sql)
            )
            results = self.cursor.fetchall()
            documents = []
            for row in results:
                try:
                    doc = json.loads(row[0])
                    documents.append(doc)
                except json.JSONDecodeError:
                    documents.append(row[0])
            return documents
        except psycopg2.Error as e:
            raise Exception(f"Failed to retrieve similar question-SQL pairs: {e}")

    def get_training_data(self, **kwargs) -> pd.DataFrame:
        try:
            self.cursor.execute(f"""
                SELECT id, question, content, training_data_type
                FROM ddl_store_{self.name_vector}
                UNION ALL
                SELECT id, question, content, training_data_type
                FROM documentation_store_{self.name_vector}
                UNION ALL
                SELECT id, question, content, training_data_type
                FROM question_sql_store_{self.name_vector}
            """)
            rows = self.cursor.fetchall()
            return pd.DataFrame(rows, columns=["id", "question", "content", "training_data_type"])
        except psycopg2.Error as e:
            raise Exception(f"Failed to retrieve training data: {e}")

    def remove_training_data(self, id: str, **kwargs) -> bool:
        try:
            if id.endswith("-sql"):
                self.cursor.execute(
                    f"DELETE FROM question_sql_store_{self.name_vector} WHERE id = %s",
                    (id,)
                )
            elif id.endswith("-ddl"):
                self.cursor.execute(
                    f"DELETE FROM ddl_store_{self.name_vector} WHERE id = %s",
                    (id,)
                )
            elif id.endswith("-doc"):
                self.cursor.execute(
                    f"DELETE FROM documentation_store_{self.name_vector} WHERE id = %s",
                    (id,)
                )
            else:
                return False
            self.conn.commit()
            return self.cursor.rowcount > 0
        except psycopg2.Error as e:
            self.conn.rollback()
            raise Exception(f"Failed to remove training data: {e}")

    def remove_collection(self, collection_name: str) -> bool:
        try:
            if collection_name == "sql":
                self.cursor.execute(f"DROP TABLE IF EXISTS question_sql_store_{self.name_vector}")
                self.cursor.execute(f"""
                    CREATE TABLE question_sql_store_{self.name_vector} (
                        id TEXT PRIMARY KEY,
                        question TEXT,
                        content TEXT NOT NULL,
                        training_data_type TEXT NOT NULL,
                        embedding vector(1536) NOT NULL
                    );
                """)
            elif collection_name == "ddl":
                self.cursor.execute(f"DROP TABLE IF EXISTS ddl_store_{self.name_vector}")
                self.cursor.execute(f"""
                    CREATE TABLE ddl_store_{self.name_vector} (
                        id TEXT PRIMARY KEY,
                        question TEXT,
                        content TEXT NOT NULL,
                        training_data_type TEXT NOT NULL,
                        embedding vector(1536) NOT NULL
                    );
                """)
            elif collection_name == "documentation":
                self.cursor.execute(f"DROP TABLE IF EXISTS documentation_store_{self.name_vector}")
                self.cursor.execute(f"""
                    CREATE TABLE documentation_store_{self.name_vector} (
                        id TEXT PRIMARY KEY,
                        question TEXT,
                        content TEXT NOT NULL,
                        training_data_type TEXT NOT NULL,
                        embedding vector(1536) NOT NULL
                    );
                """)
            else:
                return False
            self.conn.commit()
            self._create_indexes()
            return True
        except psycopg2.Error as e:
            self.conn.rollback()
            raise Exception(f"Failed to remove collection: {e}")

    def _ensure_tables_exist(self):
        try:
            self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            self.cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS ddl_store_{self.name_vector} (
                    id TEXT PRIMARY KEY,
                    question TEXT,
                    content TEXT NOT NULL,
                    training_data_type TEXT NOT NULL,
                    embedding vector(1536) NOT NULL
                );
            """)

            self.cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS documentation_store_{self.name_vector} (
                    id TEXT PRIMARY KEY,
                    question TEXT,
                    content TEXT NOT NULL,
                    training_data_type TEXT NOT NULL,
                    embedding vector(1536) NOT NULL
                );
            """)

            self.cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS question_sql_store_{self.name_vector} (
                    id TEXT PRIMARY KEY,
                    question TEXT,
                    content TEXT NOT NULL,
                    training_data_type TEXT NOT NULL,
                    embedding vector(1536) NOT NULL
                );
            """)
            self.conn.commit()
        except psycopg2.Error as e:
            self.conn.rollback()
            raise Exception(f"Failed to create tables: {e}")

    def _create_indexes(self):
        try:
            self.cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_ddl_store_{self.name_vector}_embedding
                ON ddl_store_{self.name_vector} USING hnsw (embedding vector_cosine_ops);
            """)
            self.cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_doc_store_{self.name_vector}_embedding
                ON documentation_store_{self.name_vector} USING hnsw (embedding vector_cosine_ops);
            """)
            self.cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_question_sql_store_{self.name_vector}_embedding
                ON question_sql_store_{self.name_vector} USING hnsw (embedding vector_cosine_ops);
            """)
            self.conn.commit()
        except psycopg2.Error as e:
            self.conn.rollback()
            raise Exception(f"Failed to create indexes: {e}")

class MyVanna(PGvectorCustomVectorDB, OpenAI_Chat):
    def __init__(self, config=None):
        PGvectorCustomVectorDB.__init__(self, config=config, name_vector='vanna_ia_prod')
        OpenAI_Chat.__init__(self, config=config)

def VannaConfig():
    try:
        vn = MyVanna(config={
            'api_key': os.environ.get('OPENAI_API_KEY'),
            'model': 'gpt-4o-mini',
        })
        vn.connect_to_oracle(
            dsn=os.environ.get('ORACLE_DSN'),
            user=os.environ.get('ORACLE_USER'),
            password=os.environ.get('ORACLE_PASSWORD')
        )
        return vn
    except Exception as e:
        raise Exception(f"Failed to configure Vanna: {e}")
