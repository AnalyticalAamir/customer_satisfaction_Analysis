import logging
import pandas as pd  
from zenml  import step    
from src.data_cleaning import DataCleaning , DataDivideStrategy, DataPreprocessStrategy


from typing_extensions import Annotated
from typing import Tuple
@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],

] :
    """Data cleaning class which preprocesses the data and divides it into train and test data.

    Args:
        data: pd.DataFrame
    """
    try:
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        print(df.columns)
        preprocessed_data = data_cleaning.handle_data()
        print(preprocessed_data.head)
        print(preprocessed_data.columns)

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info(f"Data Cleaning Complete")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(e)
        raise e