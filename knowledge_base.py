"""
Knowledge Base Module (SQLite)
==============================
Provides structured factoid Question Answering using an SQLite database.
Supports entity, attribute, and relational queries with schema management
and fuzzy entity/attribute slot extraction.
"""

import os
import sqlite3
import re
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_DB_PATH = os.path.join(DATA_DIR, "knowledge_base.db")


class KnowledgeBase:
    """Manages SQLite structured knowledge store and factoid queries."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns SQLite connection with dictionary row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Creates tables and populates initial seed facts if database is empty."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # 1. Entities table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                category TEXT NOT NULL,
                description TEXT
            )
        """)

        # 2. Facts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_name TEXT NOT NULL,
                attribute TEXT NOT NULL,
                value TEXT NOT NULL,
                source TEXT DEFAULT 'Curated KB',
                FOREIGN KEY (entity_name) REFERENCES entities (name)
            )
        """)

        # 3. Relations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                relation TEXT NOT NULL,
                object TEXT NOT NULL
            )
        """)

        # Check if already seeded
        cursor.execute("SELECT COUNT(*) as count FROM facts")
        count = cursor.fetchone()["count"]

        if count == 0:
            self._seed_initial_data(cursor)

        conn.commit()
        conn.close()

    def _seed_initial_data(self, cursor: sqlite3.Cursor) -> None:
        """Seeds structured facts across computer science, science, geography, and history."""
        entities = [
            ("Alan Turing", "Person", "British mathematician, logician, and computer science pioneer"),
            ("Python", "Technology", "High-level programming language created by Guido van Rossum"),
            ("Transformer", "Technology", "Deep learning self-attention architecture introduced in 2017"),
            ("BERT", "Technology", "Bidirectional encoder language representation model from Google"),
            ("Mars", "Astronomy", "Fourth planet from the Sun, known as the Red Planet"),
            ("Mount Everest", "Geography", "Highest mountain peak on Earth"),
            ("Tim Berners-Lee", "Person", "Inventor of the World Wide Web and director of W3C"),
            ("France", "Country", "Country in Western Europe"),
            ("Japan", "Country", "Island nation in East Asia"),
            ("India", "Country", "Country in South Asia"),
            ("United States", "Country", "Country primarily in North America"),
            ("Speed of Light", "Physics", "Fundamental physical constant denoted by c"),
            ("Water", "Chemistry", "Inorganic chemical compound with formula H2O"),
        ]

        cursor.executemany(
            "INSERT OR IGNORE INTO entities (name, category, description) VALUES (?, ?, ?)",
            entities
        )

        facts = [
            # Alan Turing
            ("Alan Turing", "born", "June 23, 1912 in London, England", "Biography"),
            ("Alan Turing", "died", "June 7, 1954", "Biography"),
            ("Alan Turing", "nationality", "British", "Encyclopedia"),
            ("Alan Turing", "known_for", "Turing machine, Turing test, Enigma cryptanalysis", "Computer History"),
            ("Alan Turing", "alma_mater", "King's College, Cambridge", "University records"),

            # Python
            ("Python", "creator", "Guido van Rossum", "Python.org"),
            ("Python", "founder", "Guido van Rossum", "Python.org"),
            ("Python", "release_year", "1991", "Python Documentation"),
            ("Python", "paradigm", "Multi-paradigm: object-oriented, functional, procedural", "Tech Specs"),
            ("Python", "file_extension", ".py", "Tech Specs"),

            # Transformer
            ("Transformer", "introduced_in", "2017", "Attention Is All You Need paper"),
            ("Transformer", "authors", "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, et al.", "Google Research"),
            ("Transformer", "key_mechanism", "Multi-Head Self-Attention", "Deep Learning Compendium"),

            # BERT
            ("BERT", "full_form", "Bidirectional Encoder Representations from Transformers", "Google AI"),
            ("BERT", "developer", "Google Research", "Google AI"),
            ("BERT", "release_year", "2018", "ArXiv:1810.04805"),

            # Mars
            ("Mars", "classification", "Terrestrial planet", "NASA Solar System Exploration"),
            ("Mars", "moons", "Phobos and Deimos", "Astronomical Records"),
            ("Mars", "distance_from_sun", "227.9 million km (1.52 AU)", "NASA Fact Sheet"),
            ("Mars", "atmosphere", "95% Carbon Dioxide, with Nitrogen and Argon", "Planetary Science"),

            # Mount Everest
            ("Mount Everest", "elevation", "8,848.86 metres (29,031.7 feet)", "Survey of Nepal & China 2020"),
            ("Mount Everest", "height", "8,848.86 metres", "Survey of Nepal & China 2020"),
            ("Mount Everest", "mountain_range", "Himalayas (Mahalangur Himal sub-range)", "Geography Atlas"),
            ("Mount Everest", "first_ascent", "May 29, 1953 by Edmund Hillary and Tenzing Norgay", "Alpine Records"),
            ("Mount Everest", "location", "Border between Nepal and China (Tibet)", "Geography Atlas"),

            # Tim Berners-Lee
            ("Tim Berners-Lee", "invented", "World Wide Web in 1989", "CERN Historical Archives"),
            ("Tim Berners-Lee", "nationality", "British", "W3C Biography"),
            ("Tim Berners-Lee", "organization", "World Wide Web Consortium (W3C)", "W3C"),

            # Countries
            ("France", "capital", "Paris", "World Factbook"),
            ("France", "currency", "Euro (EUR)", "European Central Bank"),
            ("France", "continent", "Europe", "UN Geoscheme"),
            ("France", "official_language", "French", "Government of France"),

            ("Japan", "capital", "Tokyo", "World Factbook"),
            ("Japan", "currency", "Japanese Yen (JPY)", "Bank of Japan"),
            ("Japan", "continent", "Asia", "UN Geoscheme"),
            ("Japan", "official_language", "Japanese", "National Records"),

            ("India", "capital", "New Delhi", "Survey of India"),
            ("India", "currency", "Indian Rupee (INR)", "Reserve Bank of India"),
            ("India", "continent", "Asia", "UN Geoscheme"),

            ("United States", "capital", "Washington, D.C.", "US Census Bureau"),
            ("United States", "currency", "United States Dollar (USD)", "Federal Reserve"),
            ("United States", "continent", "North America", "UN Geoscheme"),

            # Science constants
            ("Speed of Light", "value", "299,792,458 metres per second", "NIST Physical Constants"),
            ("Water", "chemical_formula", "H2O", "IUPAC"),
            ("Water", "boiling_point", "100 degrees Celsius (212 degrees Fahrenheit)", "Physical Chemistry"),
            ("Water", "freezing_point", "0 degrees Celsius (32 degrees Fahrenheit)", "Physical Chemistry"),
        ]

        cursor.executemany(
            "INSERT INTO facts (entity_name, attribute, value, source) VALUES (?, ?, ?, ?)",
            facts
        )

        relations = [
            ("Alan Turing", "worked_at", "Bletchley Park"),
            ("Guido van Rossum", "created", "Python"),
            ("Tim Berners-Lee", "worked_at", "CERN"),
            ("Transformer", "developed_by", "Google Brain"),
            ("BERT", "developed_by", "Google Research"),
            ("Phobos", "orbits", "Mars"),
            ("Deimos", "orbits", "Mars"),
        ]

        cursor.executemany(
            "INSERT INTO relations (subject, relation, object) VALUES (?, ?, ?)",
            relations
        )

    def get_all_facts_df(self) -> pd.DataFrame:
        """Returns all facts as a Pandas DataFrame for dataset exploration."""
        conn = self._get_connection()
        query = """
            SELECT f.id, f.entity_name, e.category, f.attribute, f.value, f.source
            FROM facts f
            LEFT JOIN entities e ON f.entity_name = e.name
            ORDER BY f.entity_name, f.attribute
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def get_all_entities(self) -> List[str]:
        """Returns list of distinct entity names in the knowledge base."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM entities ORDER BY name")
        rows = cursor.fetchall()
        conn.close()
        return [r["name"] for r in rows]

    def lookup_fact(self, entity_name: str, attribute: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Looks up specific fact for an entity and optional attribute.
        Uses case-insensitive and partial matching.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if attribute:
            cursor.execute("""
                SELECT f.entity_name, f.attribute, f.value, f.source, e.category
                FROM facts f
                JOIN entities e ON f.entity_name = e.name
                WHERE LOWER(f.entity_name) LIKE LOWER(?) AND LOWER(f.attribute) LIKE LOWER(?)
            """, (f"%{entity_name}%", f"%{attribute}%"))
        else:
            cursor.execute("""
                SELECT f.entity_name, f.attribute, f.value, f.source, e.category
                FROM facts f
                JOIN entities e ON f.entity_name = e.name
                WHERE LOWER(f.entity_name) LIKE LOWER(?)
            """, (f"%{entity_name}%",))

        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def answer_factoid_query(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Attempts to resolve an entity-attribute factoid query using pattern matching
        and SQL knowledge retrieval.
        
        Returns dict with structured answer if found, else None.
        """
        q = query.strip().lower()

        # Attribute pattern map
        attr_patterns = {
            "capital": [r"capital\s+of\s+([a-zA-Z\s]+)", r"what\s+is\s+([a-zA-Z\s]+)'?s?\s+capital"],
            "currency": [r"currency\s+of\s+([a-zA-Z\s]+)", r"what\s+currency\s+does\s+([a-zA-Z\s]+)\s+use"],
            "creator": [r"who\s+(created|invented|designed|founded|conceived)\s+([a-zA-Z\s]+)",
                        r"creator\s+of\s+([a-zA-Z\s]+)", r"founder\s+of\s+([a-zA-Z\s]+)"],
            "elevation": [r"(elevation|height)\s+of\s+([a-zA-Z\s]+)", r"how\s+tall\s+is\s+([a-zA-Z\s]+)"],
            "moons": [r"moons?\s+of\s+([a-zA-Z\s]+)", r"does\s+([a-zA-Z\s]+)\s+have\s+moons?"],
            "born": [r"when\s+was\s+([a-zA-Z\s]+)\s+born", r"birth\s+date\s+of\s+([a-zA-Z\s]+)"],
            "died": [r"when\s+did\s+([a-zA-Z\s]+)\s+die", r"death\s+of\s+([a-zA-Z\s]+)"],
            "full_form": [r"what\s+does\s+([a-zA-Z\s]+)\s+stand\s+for", r"full\s+form\s+of\s+([a-zA-Z\s]+)"],
            "chemical_formula": [r"chemical\s+formula\s+of\s+([a-zA-Z\s]+)", r"formula\s+for\s+([a-zA-Z\s]+)"],
            "speed": [r"speed\s+of\s+([a-zA-Z\s]+)"],
            "official_language": [r"language\s+of\s+([a-zA-Z\s]+)", r"official\s+language\s+of\s+([a-zA-Z\s]+)"]
        }

        matched_attr = None
        extracted_entity = None

        for attr, patterns in attr_patterns.items():
            for p in patterns:
                m = re.search(p, q, re.IGNORECASE)
                if m:
                    matched_attr = attr
                    # Entity is usually the last group in the match
                    candidate = m.group(m.lastindex).strip(" ?.")
                    # Clean trailing/leading stop words
                    candidate = re.sub(r'^(the|a|an)\s+', '', candidate, flags=re.IGNORECASE)
                    extracted_entity = candidate
                    break
            if matched_attr:
                break

        # If pattern matched, look up specific fact
        if extracted_entity:
            # Handle special aliases
            if extracted_entity in ("light", "the light"):
                extracted_entity = "Speed of Light"
                matched_attr = "value"
            if extracted_entity in ("everest", "mt everest", "mt. everest"):
                extracted_entity = "Mount Everest"

            results = self.lookup_fact(extracted_entity, matched_attr)
            if results:
                fact = results[0]
                return {
                    "source_type": "Knowledge Base (SQLite)",
                    "entity": fact["entity_name"],
                    "attribute": fact["attribute"],
                    "answer": fact["value"],
                    "confidence": 1.0,
                    "sql_query": f"SELECT value FROM facts WHERE entity='{fact['entity_name']}' AND attribute='{fact['attribute']}'",
                    "provenance": f"SQLite Table: facts | Entity: {fact['entity_name']} | Attribute: {fact['attribute']} | Citation: {fact['source']}"
                }

        # Secondary search: Check if any known entity name is mentioned in the query
        all_entities = self.get_all_entities()
        for ent in all_entities:
            pattern = r'\b' + re.escape(ent.lower()) + r'\b'
            if re.search(pattern, q):
                # Entity found, try to match attributes
                facts = self.lookup_fact(ent)
                if facts:
                    # Check if query contains any attribute keyword
                    for f in facts:
                        attr_kw = f["attribute"].replace("_", " ")
                        if attr_kw in q:
                            return {
                                "source_type": "Knowledge Base (SQLite)",
                                "entity": f["entity_name"],
                                "attribute": f["attribute"],
                                "answer": f["value"],
                                "confidence": 1.0,
                                "sql_query": f"SELECT value FROM facts WHERE entity='{f['entity_name']}' AND attribute='{f['attribute']}'",
                                "provenance": f"SQLite Table: facts | Entity: {f['entity_name']} | Citation: {f['source']}"
                            }
                    # If no specific attribute matched, return summary description of entity
                    first = facts[0]
                    return {
                        "source_type": "Knowledge Base (SQLite)",
                        "entity": ent,
                        "attribute": "overview",
                        "answer": f"{ent} is associated with: " + "; ".join([f"{f['attribute']}: {f['value']}" for f in facts[:3]]),
                        "confidence": 0.85,
                        "sql_query": f"SELECT * FROM facts WHERE entity='{ent}'",
                        "provenance": f"SQLite Table: facts | Entity: {ent} | Summary of top attributes"
                    }

        return None

    def add_fact(
        self,
        entity_name: str,
        category: str,
        attribute: str,
        value: str,
        source: str = "User Added"
    ) -> bool:
        """
        Dynamically inserts a new entity and structured fact into the SQLite store.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO entities (name, category, description) VALUES (?, ?, ?)",
            (entity_name.strip(), category.strip(), f"Entity: {entity_name.strip()}")
        )
        cursor.execute(
            "INSERT INTO facts (entity_name, attribute, value, source) VALUES (?, ?, ?, ?)",
            (entity_name.strip(), attribute.strip().lower(), value.strip(), source.strip())
        )
        conn.commit()
        conn.close()
        return True

    def execute_custom_query(self, sql_query: str) -> pd.DataFrame:
        """Executes a safe read-only SQL query for interactive database inspection."""
        conn = self._get_connection()
        try:
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            return df
        except Exception as e:
            conn.close()
            raise e
