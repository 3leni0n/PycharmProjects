import pandas as pd

from cherry import main
from glue_sessions import glue_groups
from my_fun import filter_behavior


class BehaviorData:
    def __init__(self, df):
        self.df = df

    @classmethod
    def from_cherries(cls, experiments=['2AFC_2', '2AFC_3', '2AFC_4']):
        bd = cls._load_data(experiments)
        bd._filter_behavior()
        bd._pick_cherries(experiments)
        return bd

    @classmethod
    def _load_data(cls, experiments):
        df = glue_groups(experiments=experiments, path_session='glue_sessions')
        return cls(df)

    def _filter_behavior(self):
        self.df = filter_behavior(
            self.df,
            drop_miss=True,
            clean_start=True,
            filter_drug=False
        )
        self.df.Subject = self.df.Subject.astype(int).astype(str).str.zfill(3)

    def _pick_cherries(self, experiments):
        cherries_dict = main(experiments)
        cherries = [a for expt in experiments for a in cherries_dict[expt]]
        print(f'{len(cherries)} mice: {cherries}')
        self.df = self.df[self.df['Subject'].isin(cherries)].reset_index(drop=True)