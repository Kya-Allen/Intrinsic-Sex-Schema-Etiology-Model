import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from scipy import stats
import statsmodels as sm
from factor_analyzer import FactorAnalyzer 

# parameters
SEX_SCHEMA_THRESHOLD = 1 # at what level of schematic shift will someone present clinically
HOMOSEXUALITY_THRESHOLD = 0.75 # at what level of cross-sexual orientation will one identify as homosexual
mascfem_loadings = [0.6, 0.95, 0.15, 0.14, 0.12, 0.1, 0.11, 0.14, 0.12, 0.13]

class Subject():
    def __init__(self):
        self.mascfem = 0
        self.features = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.androphilia = 0
        self.gynephilia = 0
        self.cross_schematic = False
        self.transsexual = False
        self.natal_sex = 'None'
        self.orientation_label = 'None'
        self.autogynephilia = 0
        self.autoandrophilia = 0
        self.etii_etiologic = False
        return

    def generate_feature(self, index, feature_loading):
        term_1 = feature_loading * self.mascfem
        error = np.random.randn()
        feature = term_1 + error
        self.features[index] = feature
        return
    
    def to_dict(self):
        return {
            'natal sex': self.natal_sex,
            'trans': self.transsexual,
            'orientation': self.features[0],
            'orientation label': self.orientation_label,
            'cross-schematic': self.cross_schematic,
            'latent_differentiation': self.mascfem,
            'agp': self.autogynephilia,
            'aap': self.autoandrophilia,
            'etii': self.etii_etiologic,
            'f1': self.features[2],
            'f2': self.features[3],
            'f3': self.features[4],
            'f4': self.features[5],
            'f5': self.features[6],
            'f6': self.features[7],
            'f7': self.features[8],
            'f8': self.features[9]
        }
    
    def determine_agp(self):
        if self.features[0] < -1.25:
            self.autogynephilia = 0
            return
        self.autogynephilia = np.random.exponential(1.75)
        return
        
    def determine_aap(self):
        if self.features[0] > 1.25:
            self.autoandrophilia = 0
            return
        self.autoandrophilia = np.random.exponential(1.75)
        return
    
class Masc(Subject):
    def __init__(self):
        super().__init__()
        self.natal_sex = 'Male'
        self.mascfem = np.random.normal(4, 1.95)

        for index, loading in enumerate(mascfem_loadings):
            self.generate_feature(index, loading)
        self.determine_aap()
        self.determine_agp()
        self.determine_etii()
        self.determine_cross_schema()

        if self.features[0] > HOMOSEXUALITY_THRESHOLD: self.orientation_label = 'Straight Male'
        elif self.features[0] < -HOMOSEXUALITY_THRESHOLD: self.orientation_label = 'Gay Male'
        else: self.orientation_label = 'Bi Male'
        return
    
    def determine_cross_schema(self):
        if self.features[1] < -SEX_SCHEMA_THRESHOLD:
            self.cross_schematic = True
            self.transsexual = True

    def determine_etii(self):
        multiplier = 1 + (self.autogynephilia ** self.autogynephilia)
        base_rate = 0.00000001
        p = base_rate * multiplier
        if p > 0.3: p = 0.3
        if np.random.binomial(1, p):
            self.etii_etiologic = True
            self.transsexual = True
        return

class Fem(Subject):
    def __init__(self):
        super().__init__()
        self.natal_sex = 'Female'
        self.mascfem = np.random.normal(-4, 1.95)

        for index, loading in enumerate(mascfem_loadings):
            self.generate_feature(index, loading)
        self.determine_aap()
        self.determine_agp()
        self.determine_etii()
        self.determine_cross_schema()
        if self.features[0] > HOMOSEXUALITY_THRESHOLD: self.orientation_label = 'Gay Female'
        elif self.features[0] < -HOMOSEXUALITY_THRESHOLD: self.orientation_label = 'Straight Female'
        else: self.orientation_label = 'Bi Female'
        return
    
    def determine_cross_schema(self):
        if self.features[1] > SEX_SCHEMA_THRESHOLD:
            self.cross_schematic = True
            self.transsexual = True

    def determine_etii(self):
        multiplier = 1 + (self.autoandrophilia ** self.autoandrophilia)
        base_rate = 0.00000001
        p = base_rate * multiplier
        if p > 0.3: p = 0.3
        if np.random.binomial(1, p):
            self.etii_etiologic = True
            self.transsexual = True
        return