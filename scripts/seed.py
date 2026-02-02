import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import pandas as pd
from pymongo import MongoClient
from sqlalchemy import create_engine


class DataSeeder:
    def __init__(self):
        self.files = [
            "albums.parquet",
            "album_images.parquet",
            "artist_album.parquet",
            "artist_genre.parquet",
            "artist_image.parquet",
            "artists.parquet",
            "available_markets.parquet",
            "track_artist.parquet",
            "tracks.parquet",
        ]

        self.mongo_client = MongoClient(
            "mongodb://mongodb:mongodb@localhost:27017/?authSource=admin"
        )
        self.mongo_db = self.mongo_client["mongo_store"]

        self.mysql_engine = create_engine(
            "mysql+mysqlconnector://mysql:mysql@localhost:3306/mysql_store"
        )

        self.postgres_engine = create_engine(
            "postgresql+psycopg2://postgres:postgres@localhost:5432/postgres"
        )

        os.makedirs("./assets/sqlite", exist_ok=True)

        self.sqlite_engine = create_engine("sqlite:///assets/sqlite/nisaba.sqlite")

    def to_csv(self, df: pd.DataFrame, filename: str) -> Dict:
        """Write DataFrame to CSV"""
        try:
            output_path = f"assets/csv/{filename.replace('.parquet', '.csv')}"

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            df.to_csv(output_path, index=False)

            return {"destination": "CSV", "status": "success", "file": filename}
        except Exception as e:
            report = {
                "destination": "CSV",
                "status": "failed",
                "file": filename,
                "error": str(e),
            }
            raise Exception(report)

    def to_excel(self, df: pd.DataFrame, filename: str) -> Dict:
        """Write DataFrame to Excel"""
        try:
            output_path = f"assets/xlsx/{filename.replace('.parquet', '.xlsx')}"

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            sheet_name = filename.split(".")[0]

            df.to_excel(output_path, sheet_name=sheet_name, index=False)

            return {"destination": "Excel", "status": "success", "file": filename}
        except Exception as e:
            report = {
                "destination": "Excel",
                "status": "failed",
                "file": filename,
                "error": str(e),
            }
            raise Exception(report)

    def to_mongo(self, df: pd.DataFrame, filename: str) -> Dict:
        """Write DataFrame to MongoDB"""
        try:
            df_copy = df.copy()
            collection_name = filename.split(".")[0]
            collection = self.mongo_db[collection_name]

            # Convert date columns to string
            if "release_date" in df_copy.columns:
                df_copy["release_date"] = df_copy["release_date"].astype("string")

            data_dict = df_copy.to_dict(orient="records")
            collection.delete_many({})  # Clear existing data

            collection.insert_many(data_dict)

            return {
                "destination": "MongoDB",
                "status": "success",
                "file": filename,
                "records": len(data_dict),
            }
        except Exception as e:
            report = {
                "destination": "MongoDB",
                "status": "failed",
                "file": filename,
                "error": str(e),
            }
            raise Exception(report)

    def to_mysql(self, df: pd.DataFrame, filename: str) -> Dict:
        """Write DataFrame to MySQL"""
        try:
            table_name = filename.split(".")[0]

            df.to_sql(
                name=table_name, con=self.mysql_engine, if_exists="replace", index=False
            )

            return {"destination": "MySQL", "status": "success", "file": filename}
        except Exception as e:
            report = {
                "destination": "MySQL",
                "status": "failed",
                "file": filename,
                "error": str(e),
            }
            raise Exception(report)

    def to_postgres(self, df: pd.DataFrame, filename: str) -> Dict:
        """Write DataFrame to PostgreSQL"""
        try:
            table_name = filename.split(".")[0]
            df.to_sql(
                name=table_name,
                con=self.postgres_engine,
                if_exists="replace",
                index=False,
            )

            return {"destination": "PostgreSQL", "status": "success", "file": filename}
        except Exception as e:
            report = {
                "destination": "PostgreSQL",
                "status": "failed",
                "file": filename,
                "error": str(e),
            }
            raise Exception(report)

    def to_sqlite(self, df: pd.DataFrame, filename: str):
        try:
            table_name = filename.split(".")[0]
            df.to_sql(
                name=table_name,
                con=self.sqlite_engine,
                if_exists="replace",
                index=False,
            )
        except Exception as e:
            report = {
                "destination": "Sqlite",
                "status": "failed",
                "file": filename,
                "error": str(e),
            }
            raise Exception(report)

    def seed_file_parallel(self, filename: str, destinations: List[str]) -> List[Dict]:
        """
        Read a file once and write to multiple destinations in parallel

        Args:
            filename: Name of the parquet file to process
            destinations: List of destinations ('csv', 'excel', 'mongo', 'mysql', 'postgres', 'sqlite')
                         If None, writes to all destinations
        """
        if len(destinations) == 0:
            destinations = ["csv", "excel", "mongo", "mysql", "postgres", "sqlite"]

        # Read once
        try:
            df = pd.read_parquet(f"assets/parquet/{filename}")

        except Exception as e:
            report = [{"status": "failed", "error": f"Read failed: {e}"}]
            raise Exception(report)

        # Map destinations to methods
        destination_map = {
            "csv": self.to_csv,
            "excel": self.to_excel,
            "mongo": self.to_mongo,
            "mysql": self.to_mysql,
            "postgres": self.to_postgres,
            "sqlite": self.to_sqlite,
        }

        results = []

        # Write to multiple destinations in parallel
        with ThreadPoolExecutor(max_workers=len(destinations)) as executor:
            futures = {
                executor.submit(destination_map[dest], df.copy(), filename): dest
                for dest in destinations
                if dest in destination_map
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    dest = futures[future]

                    results.append(
                        {
                            "destination": dest,
                            "status": "failed",
                            "file": filename,
                            "error": str(e),
                        }
                    )

        return results

    def seed_all_files(self, destinations: List[str] = [], max_workers: int = 4):
        """
        Process all files, each with parallel writes to destinations

        Args:
            destinations: List of destinations to write to
            max_workers: Maximum number of files to process concurrently
        """

        all_results = []

        # Process files with limited concurrency to avoid overwhelming the system
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.seed_file_parallel, file, destinations): file
                for file in self.files
            }

            for future in as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    raise Exception(f"Failed to process {filename}: {e}")

        print("Seeding Complete!")

        return all_results


def main():
    seeder = DataSeeder()

    try:
        seeder.seed_all_files(max_workers=4)

    except Exception as e:
        raise Exception(str(e))


if __name__ == "__main__":
    main()
