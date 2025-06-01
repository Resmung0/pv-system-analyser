import pandas as pd

class ClassificationSplitter:
    """
    Splits a classification dataset into train, validation and test sets.

    Attributes:
        data (pd.DataFrame): The input dataset to split.

    Methods:
        sample_data: Samples a fraction of rows from the dataset.
        split_dataset: Splits the dataset into train, validation and test sets.
    """
    def __init__(self, dataset: pd.DataFrame) -> None:
        """
        Initializes the splitter with the input dataset.

        Args:
            dataset (pd.DataFrame): The input dataset to split.
        """
        self.data = dataset
    
    @staticmethod
    def sample_data(
        data: pd.DataFrame, column: str, fraction: float, stratify: bool
    ) -> pd.DataFrame:
        """
        Samples a fraction of rows from the dataset.

        Args:
            data (pd.DataFrame): Data for sample.
            column (str): The column based the sampling when stratify is true. 
            fraction (float): The fraction of rows to sample.
            stratify (bool): Whether to stratify the sampling.

        Returns:
            pd.DataFrame: The sampled rows.
        """
        if stratify:
            data = data.groupby(column)
            data = data.apply(lambda x: x.sample(frac=fraction))
            return data.droplevel(0)
        return data.sample(frac=fraction)
    
    def __call__(
        self,
        class_column: str,
        train_size: float,
        val_size: float,
        test_size: float,
        stratify: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the dataset into train, validation and test sets.

        Args:
            class_column (str): The class column to split on.
            train_size (float): The fraction of rows for training set.
            val_size (float): The fraction of rows for validation set.
            test_size (float): The fraction of rows for test set.
            stratify (bool): Whether to stratify the splitting. Default False.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The split train, 
                validation and test sets.
        """
        train_dataset = self.sample_data(self.data, class_column, train_size, stratify)
    
        data = self.data.drop(train_dataset.index)
        valid_dataset = self.sample_data(data, class_column, val_size, stratify)
        
        data = data.drop(valid_dataset.index)
        test_dataset = self.sample_data(data, class_column, test_size, stratify)        
        
        return train_dataset, valid_dataset, test_dataset      
