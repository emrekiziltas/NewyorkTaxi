import os
import calendar
from datetime import datetime
from pathlib import Path

from source.core.downloader import DataDownloader
from source.core.validator import DataValidator
from source.core.transformer import DataTransformer
from source.enrichment.weather import WeatherEnricher
from source.utils.logger import logger
from source.config.settings import Config


def main():
    """Main pipeline execution function."""
    start_time = datetime.now()

    logger.info("\n" + "=" * 60)
    logger.info("NYC TAXI DATA PIPELINE")
    logger.info("=" * 60 + "\n")

    # Configuration
    YEARS = range(2023, 2024)
    MONTHS = range(1, 3)
    TAXI_TYPE = "yellow"
    ADD_WEATHER = True  # Set to True to add weather data

    # Initialize components
    downloader = DataDownloader(
        raw_dir=Config.DEFAULT_RAW_DIR,
        processed_dir=Config.DEFAULT_PROCESSED_DIR
    )
    validator = DataValidator()
    transformer = DataTransformer()
    weather_enricher = WeatherEnricher()

    # Ensure borough lookup is saved
    downloader.save_borough_data()

    processed_files, failed_files = [], []

    for year in YEARS:
        for month in MONTHS:
            try:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Processing: {TAXI_TYPE.upper()} Taxi - {year}-{month:02d}")
                logger.info(f"{'=' * 60}")

                # Step 1: Download
                logger.info(f"Step 1: Download")
                df = downloader.download_parquet_file(year, month, TAXI_TYPE)

                if df is None or df.empty:
                    failed_files.append(f"{year}-{month:02d}")
                    logger.warning(f"No data for {year}-{month:02d}")
                    continue

                # Step 2: Validate
                logger.info(f"Step 2: Validate")
                quality_report = validator.validate_data_quality(df)
                if not quality_report["passed"]:
                    logger.error(f"Data quality check failed: {quality_report.get('reason', 'Unknown')}")
                    failed_files.append(f"{year}-{month:02d}")
                    continue
                logger.info(f"Step 3: Transform")
                # Step 3: Transform
                df_transformed = transformer.transform_data(df)

                if df_transformed is None or df_transformed.empty:
                    logger.error(f"Transformation resulted in empty DataFrame for {year}-{month:02d}")
                    failed_files.append(f"{year}-{month:02d}")
                    continue

                logger.info(f"Step 4: Add borough info")
                # Step 4: Add borough info
                current_dir = Path(os.path.dirname(__file__))
                lookup_path = current_dir / Config.DEFAULT_LOOKUP_DIR / "taxi_zone_lookup.csv"
                df_transformed = transformer.add_borough_info(df_transformed, lookup_path)

                logger.info(f"Step 5: Enrich with weather (optional)")
                # Step 5: Enrich with weather (optional)
                if ADD_WEATHER:
                    try:
                        logger.info("Adding weather data...")
                        start_date = f"{year}-{month:02d}-01"
                        # Calculate last day of month
                        last_day = calendar.monthrange(year, month)[1]
                        end_date = f"{year}-{month:02d}-{last_day}"

                        weather_df = weather_enricher.fetch_weather_data(start_date, end_date)
                        df_transformed = weather_enricher.add_weather_info(df_transformed, weather_df)
                    except Exception as e:
                        logger.warning(f"Failed to add weather data: {e}")
                        # Continue without weather data

                # Step 6: Save processed file
                processed_file_path = downloader.processed_dir / f"{TAXI_TYPE}_{year}_{month:02d}_processed.parquet"
                df_transformed.to_parquet(processed_file_path, index=False)
                processed_files.append(processed_file_path)
                logger.info(f"Successfully processed: {processed_file_path}")

            except Exception as e:
                logger.error(f"Failed to process {year}-{month:02d}: {e}", exc_info=True)
                failed_files.append(f"{year}-{month:02d}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)

    logger.info(f"Successfully processed: {len(processed_files)} files")
    if failed_files:
        logger.warning(f"Failed to process: {len(failed_files)} files")
        logger.warning(f"Failed periods: {', '.join(failed_files)}")
    else:
        logger.info("All files processed successfully!")

    elapsed_time = datetime.now() - start_time
    logger.info(f"\n Total execution time: {elapsed_time}")
    logger.info("=" * 60 + "\n")

    # Cleanup
    downloader.close()


if __name__ == "__main__":
    import sys

    # Fix Unicode logging issue on Windows
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nPipeline failed: {e}", exc_info=True)
        sys.exit(1)