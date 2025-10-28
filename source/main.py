import pandas as pd

from utils.logger import logger
from config.settings import config
from data.downloader import DataDownloader


def main():
    """Main pipeline execution function."""
    start_time = datetime.now()

    logger.info("\n" + "=" * 60)
    logger.info("NYC TAXI DATA PIPELINE")
    logger.info("=" * 60 + "\n")

    pipeline = NYCTaxiDataIngestion(raw_dir="data/raw", processed_dir="data/processed")


    YEARS = range(2023, 2025)
    MONTHS = range(1, 13)
    TAXI_TYPE = "yellow"

    processed_files, failed_files = [], []

    for year in YEARS:
        for month in MONTHS:
            result = pipeline.process_month(year, month, TAXI_TYPE)
            if result:
                processed_files.append(result)
            else:
                failed_files.append(f"{year}-{month:02d}")

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)

    logger.info(f"[OK] Successfully processed: {len(processed_files)} files")
    logger.error(f"[X] Failed to process some files")
    if failed_files:
        logger.warning(f"✗ Failed to process: {len(failed_files)} files")
        logger.warning(f"  Failed periods: {', '.join(failed_files)}")

    weather_df = fetch_weather_data("2024-01-01", "2024-12-31")

    elapsed_time = datetime.now() - start_time
    logger.info(f"\nTotal execution time: {elapsed_time}")
    logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n⚠ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)