import argparse
from pathlib import Path
import time
import lance
import duckdb
import pyarrow
import pyarrow.dataset
import os
import logging
import matplotlib.pyplot as plt


def prepare_dataset(scalefactor: int, path: Path):
    full_path = path / f"sf{scalefactor}"
    if os.path.exists(full_path):
        logging.info("Dataset already created, skipping")
        return

    os.makedirs(full_path / "parquet")
    os.makedirs(full_path / "lance")

    logging.info("Generating TPC-H data with scalefactor=%d", scalefactor)
    db = duckdb.connect()
    db.sql(f"CALL dbgen(sf = {scalefactor})")

    logging.info(f"Exporting TPC-H data.")

    for tablename in [
        "customer",
        "lineitem",
        "nation",
        "orders",
        "part",
        "partsupp",
        "region",
        "supplier",
    ]:
        table = db.sql(f"select * from {tablename}")

        # parquet
        table.write_parquet((full_path / "parquet" / f"{tablename}.parquet").as_posix())

        # lance
        lance.write_dataset(table.to_arrow_table(), (full_path / "lance" / f"{tablename}.lance"))

    logging.info("Dataset created.")


def run_queries(runs: int) -> dict[str, float]:
    times = {}

    for i in range(1, 23):
        logging.debug("Running query %d", i)
        start = time.time()
        for _ in range(runs):
            duckdb.sql(f"PRAGMA tpch({i})").execute()
        times[i] = (time.time() - start) / runs

    return times


def register_parquet(dataset_path: Path) -> None:
    duckdb.register("customer", pyarrow.dataset.dataset(dataset_path / "customer.parquet"))
    duckdb.register("lineitem", pyarrow.dataset.dataset(dataset_path / "lineitem.parquet"))
    duckdb.register("nation", pyarrow.dataset.dataset(dataset_path / "nation.parquet"))
    duckdb.register("orders", pyarrow.dataset.dataset(dataset_path / "orders.parquet"))
    duckdb.register("part", pyarrow.dataset.dataset(dataset_path / "part.parquet"))
    duckdb.register("partsupp", pyarrow.dataset.dataset(dataset_path / "partsupp.parquet"))
    duckdb.register("region", pyarrow.dataset.dataset(dataset_path / "region.parquet"))
    duckdb.register("supplier", pyarrow.dataset.dataset(dataset_path / "supplier.parquet"))


def register_lance(dataset_path: Path) -> None:
    duckdb.register("customer", lance.dataset(dataset_path / "customer.lance"))
    duckdb.register("lineitem", lance.dataset(dataset_path / "lineitem.lance"))
    duckdb.register("nation", lance.dataset(dataset_path / "nation.lance"))
    duckdb.register("orders", lance.dataset(dataset_path / "orders.lance"))
    duckdb.register("part", lance.dataset(dataset_path / "part.lance"))
    duckdb.register("partsupp", lance.dataset(dataset_path / "partsupp.lance"))
    duckdb.register("region", lance.dataset(dataset_path / "region.lance"))
    duckdb.register("supplier", lance.dataset(dataset_path / "supplier.lance"))


def main():
    parser = argparse.ArgumentParser(description="TPCH Benchmark")
    parser.add_argument("-r", "--runs", type=int, default=1, help="Number of runs per query")
    parser.add_argument(
        "-s", "--scalefactor", type=int, default=1, help="Scale of the TPC-H dataset"
    )
    parser.add_argument(
        "-d", "--dataset", type=Path, default="./dataset", help="Path to the dataset"
    )
    parser.add_argument("-l", "--logging_level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(level=args.logging_level)

    # run

    prepare_dataset(args.scalefactor, args.dataset)

    logging.info("Running Parquet")
    register_parquet(args.dataset / f"sf{args.scalefactor}" / "parquet")
    parquet_times = run_queries(args.runs)

    logging.info("Running Lance")
    register_lance(args.dataset / f"sf{args.scalefactor}" / "lance")
    lance_times = run_queries(args.runs)

    # output

    parquet_times = pyarrow.table({"query": parquet_times.keys(), "rt": parquet_times.values()})
    lance_times = pyarrow.table({"query": lance_times.keys(), "rt": lance_times.values()})
    individual = duckdb.sql(
        """
        select p.query, round(p.rt, 3) as parquet, round(l.rt, 3) as lance,
            'x' || round(l.rt / p.rt, 3) as diff
        from parquet_times p
        join lance_times l on p.query = l.query
        order by 1
        """
    )
    aggregated = duckdb.sql(
        """
        select round(sum(p.rt), 3) as parquet, round(sum(l.rt), 3) as lance,
            'x' || round(sum(l.rt) / sum(p.rt), 3) as diff
        from parquet_times p
        join lance_times l on p.query = l.query
        """
    )

    print("Time per query")
    individual.show()

    print("Aggregated time")
    aggregated.show()

    individual.write_csv(f"sf{args.scalefactor}.csv")

    ax = individual.to_df().plot(
        x="query",
        y=["parquet", "lance"],
        kind="bar",
        color=["#333", "#e69353"],
        width=0.6,
        figsize=(8, 3),
    )
    ax.set_axisbelow(True)
    plt.grid(color="#eee", linewidth=1)
    plt.title(f"TPC-H Performance (sf={args.scalefactor})")
    plt.xlabel("Query")
    plt.ylabel("Response time (s)")
    plt.tight_layout()
    plt.savefig(f"sf{args.scalefactor}.png")


if __name__ == "__main__":
    main()
