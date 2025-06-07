import pandas as pd
from sklearn.preprocessing import StandardScaler

class GHIDataPreprocessor:
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.df = None
        self.scaled_data = None
        self.features = None

    def load_and_merge(self):
        # Load tiap file dan pilih kolom Country + 2022 saja
        df_undernourishment = pd.read_csv(self.file_paths['undernourishment'], encoding='latin1')
        df_undernourishment = df_undernourishment[['Country', '2022']]
        df_undernourishment.rename(columns={'2022': "Undernourishment"}, inplace=True)

        df_stunting = pd.read_csv(self.file_paths['stunting'], encoding='latin1')
        df_stunting = df_stunting[['Country', '2022']]
        df_stunting.rename(columns={'2022': "Stunting"}, inplace=True)

        df_wasting = pd.read_csv(self.file_paths['wasting'], encoding='latin1')
        df_wasting = df_wasting[['Country', '2022']]
        df_wasting.rename(columns={'2022': "Wasting"}, inplace=True)

        df_mortality = pd.read_csv(self.file_paths['mortality'], encoding='latin1')
        df_mortality = df_mortality[['Country', '2022']]
        df_mortality.rename(columns={'2022': "Mortality"}, inplace=True)

        df_ghi = pd.read_csv(self.file_paths['ghi'], encoding='latin1')
        df_ghi = df_ghi[['Country', '2022']]
        df_ghi.rename(columns={'2022': 'GHI Score'}, inplace=True)


        # Merge semua data berdasarkan 'Country'
        df = df_ghi.merge(df_undernourishment, on='Country')
        df = df.merge(df_stunting, on='Country')
        df = df.merge(df_wasting, on='Country')
        df = df.merge(df_mortality, on='Country')

        self.df = df.dropna()
        self.features = ["Undernourishment", "Stunting", "Wasting", "Mortality", "GHI Score"]

    def scale_features(self):
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(self.df[self.features])
