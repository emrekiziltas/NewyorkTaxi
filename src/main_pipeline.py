import requests
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NYCTaxiDataIngestion:
    """NYC Taxi Trip verilerini çeken ve işleyen sınıf"""

    def __init__(self, data_dir='data/raw'):
        """
        Args:
            data_dir: Ham verinin kaydedileceği dizin
        """
        self.base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_parquet_file(self, year, month, taxi_type='yellow'):
        """
        Belirtilen ay için NYC Taxi verisi indir

        Args:
            year: Yıl (örn: 2024)
            month: Ay (1-12)
            taxi_type: 'yellow', 'green', veya 'fhv'

        Returns:
            DataFrame veya None
        """
        try:
            filename = f"{taxi_type}_tripdata_{year}-{month:02d}.parquet"
            url = f"{self.base_url}/{filename}"
            local_path = self.data_dir / filename

            # Eğer dosya zaten varsa, doğrudan oku
            if local_path.exists():
                logger.info(f"Dosya zaten mevcut: {local_path}. Okunuyor...")
                return pd.read_parquet(local_path)

            logger.info(f"İndiriliyor: {url}")

            # Dosyayı indir
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            # Dosyayı kaydet
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"İndirme tamamlandı: {local_path}")

            # DataFrame olarak oku
            df = pd.read_parquet(local_path)
            logger.info(f"Veri yüklendi: {len(df):,} kayıt")

            return df

        except requests.exceptions.HTTPError as e:
            logger.warning(f"Veri indirilemedi (muhtemelen mevcut değil): {url} - Hata: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"İndirme sırasında ağ hatası: {e}")
            return None
        except Exception as e:
            logger.error(f"Veri işlenirken beklenmeyen hata: {e}")
            return None

    def basic_data_quality_checks(self, df):
        """
        Temel veri kalite kontrolleri

        Args:
            df: Kontrol edilecek DataFrame

        Returns:
            dict: Veri kalite raporu
        """
        if df is None or df.empty:
            return {"status": "failed", "reason": "Empty DataFrame"}

        report = {
            "total_records": len(df),
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_records": df.duplicated().sum(),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}
        }

        # Tarih sütunları için kontrol
        date_columns = [col for col in df.columns if 'datetime' in col.lower()]
        if date_columns:
            for col in date_columns:
                try:
                    report[f"{col}_range"] = {
                        "min": str(df[col].min()),
                        "max": str(df[col].max())
                    }
                except Exception as e:
                    logger.warning(f"{col} sütunu için min/max alınamadı: {e}")
                    report[f"{col}_range"] = "Hesaplanamadı"

        logger.info("Veri kalite kontrolü tamamlandı")
        return report

    def basic_transformation(self, df):
        """
        Temel veri dönüşümleri

        Args:
            df: Dönüştürülecek DataFrame

        Returns:
            DataFrame: Dönüştürülmüş veri
        """
        if df is None or df.empty:
            return df

        logger.info("Temel dönüşümler uygulanıyor...")

        df_transformed = df.copy()

        # Tarih sütunlarını datetime'a çevir
        date_columns = [col for col in df_transformed.columns if 'datetime' in col.lower()]
        for col in date_columns:
            if not pd.api.types.is_datetime64_any_dtype(df_transformed[col]):
                logger.info(f"{col} sütunu datetime formatına çevriliyor...")
                df_transformed[col] = pd.to_datetime(df_transformed[col], errors='coerce')

        # Negatif değerleri temizle
        numeric_cols_to_check = [
            'trip_distance', 'fare_amount', 'total_amount',
            'tip_amount', 'tolls_amount'
        ]

        for col in numeric_cols_to_check:
            if col in df_transformed.columns and pd.api.types.is_numeric_dtype(df_transformed[col]):
                initial_count = len(df_transformed)
                df_transformed = df_transformed[df_transformed[col] >= 0]
                removed_count = initial_count - len(df_transformed)
                if removed_count > 0:
                    logger.info(f"{col} sütunundaki {removed_count} negatif kayıt kaldırıldı.")

        # Tarih bilgilerini ekle
        pickup_col = None
        if 'tpep_pickup_datetime' in df_transformed.columns:
            pickup_col = 'tpep_pickup_datetime'
        elif 'lpep_pickup_datetime' in df_transformed.columns:
            pickup_col = 'lpep_pickup_datetime'

        if pickup_col and pd.api.types.is_datetime64_any_dtype(df_transformed[pickup_col]):
            logger.info(f"{pickup_col} üzerinden tarih özellikleri çıkarılıyor...")
            df_transformed['pickup_hour'] = df_transformed[pickup_col].dt.hour
            df_transformed['pickup_day'] = df_transformed[pickup_col].dt.day
            df_transformed['pickup_weekday'] = df_transformed[pickup_col].dt.dayofweek
            df_transformed['pickup_month'] = df_transformed[pickup_col].dt.month
            df_transformed['pickup_year'] = df_transformed[pickup_col].dt.year
        else:
            logger.warning("pickup_datetime sütunu bulunamadı veya datetime formatında değil.")

        logger.info(f"Dönüşüm tamamlandı: {len(df_transformed):,} kayıt")
        return df_transformed

    def save_to_parquet(self, df, output_path):
        """
        DataFrame'i Parquet olarak kaydet

        Args:
            df: Kaydedilecek DataFrame
            output_path: Çıktı dosya yolu
        """
        try:
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)

            # DÜZELTME: Girinti düzeltildi
            df.to_parquet(output_path_obj, index=False, compression='snappy')
            logger.info(f"Veri Parquet olarak kaydedildi: {output_path_obj}")

        except Exception as e:
            logger.error(f"Parquet kaydetme hatası: {e}")

    def process_month(self, year, month, taxi_type='yellow'):

        logger.info(f"\n--- {year}-{month:02d} verisi işleniyor ---")

        # İndir
        df = self.download_parquet_file(year=year, month=month, taxi_type=taxi_type)

        if df is None or df.empty:
            logger.warning(f"{year}-{month:02d} için veri bulunamadı.")
            return None

        # Dönüştür
        df_transformed = self.basic_transformation(df)

        if df_transformed is None or df_transformed.empty:
            logger.warning(f"{year}-{month:02d} dönüşüm başarısız.")
            return None

        # Kaydet
        output_path = Path(f'data/processed/{taxi_type}_taxi_{year}_{month:02d}_processed.parquet')
        self.save_to_parquet(df_transformed, output_path)

        return output_path

def merge_parquet_files(output_path="data/processed/yellow_taxi_2024_merged.parquet"):
    """İşlenmiş 12 aylık dosyayı birleştirir"""
    from pathlib import Path
    import pandas as pd
    import logging

    processed_dir = Path("data/processed")
    parquet_files = sorted(processed_dir.glob("yellow_taxi_2024_*_processed.parquet"))

    if not parquet_files:
        logging.error("⚠️ Hiç işlenmiş Parquet dosyası bulunamadı.")
        return

    logging.info(f"🔍 {len(parquet_files)} dosya bulundu. Birleştirme başlıyor...")

    dfs = []
    for f in parquet_files:
        logging.info(f"Yükleniyor: {f.name}")
        dfs.append(pd.read_parquet(f))

    merged_df = pd.concat(dfs, ignore_index=True)
    logging.info(f"✅ Toplam {len(merged_df):,} satır birleştirildi.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(output_path, index=False, compression="snappy")

    logging.info(f"💾 Birleştirilmiş veri kaydedildi: {output_path}")
    return output_path


def main():
    """Ana pipeline fonksiyonu"""

    logger.info("=" * 60)
    logger.info("NYC Taxi Data Pipeline Başlatılıyor")
    logger.info("=" * 60)

    ingestion = NYCTaxiDataIngestion(data_dir='data/raw')

    processed_files = []
    for year in range (2023, 2025):
        for month in  range (1, 13):
             output_path = ingestion.process_month(year=year, month=month)
             if output_path:
               processed_files.append(output_path)

        if processed_files:
            logger.info(f"\n{len(processed_files)} ay başarıyla işlendi.")
        else:
            logger.error("Hiçbir veri işlenemedi.")

        if processed_files:
            logger.info(f"\n{len(processed_files)} ay başarıyla işlendi.")
            merged_file = merge_parquet_files()
            if merged_file:
                logger.info(f"🎉 Tüm aylar başarıyla birleştirildi: {merged_file}")
        else:
            logger.error("Hiçbir veri işlenemedi.")


if __name__ == "__main__":
    main()