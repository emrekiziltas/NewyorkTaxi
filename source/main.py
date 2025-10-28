from datetime import datetime
from source.utils.logger import logger
from source.config.settings import Config
from source.core.downloader import DataDownloader
from source.core.validator import DataValidator
from source.core.transformer import DataTransformer


def main():
    """Main pipeline execution function."""
    start_time = datetime.now()

    logger.info("\n" + "=" * 60)
    logger.info("NYC TAXI DATA PIPELINE")
    logger.info("=" * 60 + "\n")

    # Initialize the downloader
    downloader = DataDownloader(raw_dir=Config.DEFAULT_RAW_DIR, processed_dir="data/processed")

    YEARS = range(2023, 2025)
    MONTHS = range(1, 3)
    TAXI_TYPE = "yellow"

    processed_files, failed_files = [], []
    validator = DataValidator()
    transformer = DataTransformer()


    for year in YEARS:
        for month in MONTHS:
            try:
                df = downloader.download_parquet_file(year, month, TAXI_TYPE)

                if df is None or df.empty:
                    failed_files.append(f"{year}-{month:02d}")
                    logger.warning(f"No data for {year}-{month:02d}")
                    continue

                # Step 1: Validate
                quality_report = validator.validate_data_quality(df)
                if not quality_report["passed"]:
                    logger.error(f"Data quality check failed: {quality_report['reason']}")
                    failed_files.append(f"{year}-{month:02d}")
                    continue

                # Step 2: Transform
                df_transformed = transformer.transform_data(df)
                if df_transformed is None or df_transformed.empty:
                    logger.error(f"Transformation resulted in empty DataFrame for {year}-{month:02d}")
                    failed_files.append(f"{year}-{month:02d}")
                    continue

                # Step 3: Save processed file
                processed_file_path = downloader.processed_dir / f"{TAXI_TYPE}_{year}_{month:02d}_processed.parquet"
                df_transformed.to_parquet(processed_file_path)
                processed_files.append(processed_file_path)
                logger.info(f"Successfully processed: {processed_file_path}")

            except Exception as e:
                logger.error(f"Failed to process {year}-{month:02d}: {e}", exc_info=True)
                failed_files.append(f"{year}-{month:02d}")

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)

    logger.info(f"[OK] Successfully processed: {len(processed_files)} files")
    if failed_files:
        logger.warning(f"[X] Failed to process: {len(failed_files)} files")
        logger.warning(f"  Failed periods: {', '.join(failed_files)}")

    elapsed_time = datetime.now() - start_time
    logger.info(f"\nTotal execution time: {elapsed_time}")
    logger.info("=" * 60 + "\n")

    downloader.close()

if __name__ == "__main__":
    import sys
    # Fix Unicode logging issue on Windows
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
    except Exception as e:
        logger.error(f"\nPipeline failed: {e}", exc_info=True)
