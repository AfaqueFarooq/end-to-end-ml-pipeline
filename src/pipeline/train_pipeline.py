import sys
from src.exceptions import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            # Step 1 - Data Ingestion
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion()
            train_path, test_path = data_ingestion.initiate_data_ingestion()

            # Step 2 - Data Transformation
            logging.info("Starting data transformation")
            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)

            # Step 3 - Model Training
            logging.info("Starting model training")
            model_trainer = ModelTrainer()
            r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
            logging.info(f"Training complete. R2 Score: {r2_score}")
            print(f"Training complete. R2 Score: {r2_score}")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()