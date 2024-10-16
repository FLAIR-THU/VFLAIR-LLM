import sys, os

sys.path.append(os.pardir)

from evaluates.attacks.AttributeInference import AttributeInference
from evaluates.attacks.BatchLabelReconstruction import BatchLabelReconstruction

from evaluates.attacks.DataReconstruct import DataReconstruction
from evaluates.attacks.DirectionbasedScoring import DirectionbasedScoring
from evaluates.attacks.NormbasedScoring import NormbasedScoring
from evaluates.attacks.NoisyLabel import NoisyLabel
from evaluates.attacks.ModelCompletion import ModelCompletion
from evaluates.attacks.DirectLabelScoring import DirectLabelScoring
from evaluates.attacks.GenerativeRegressionNetwork import GenerativeRegressionNetwork
from evaluates.attacks.ResSFL import ResSFL
from evaluates.attacks.ASB import ASB

# LLM attacks
from evaluates.attacks.VanillaModelInversion_WhiteBox import VanillaModelInversion_WhiteBox
from evaluates.attacks.VanillaModelInversion_WhiteBox_mse import VanillaModelInversion_WhiteBox_mse

from evaluates.attacks.VanillaModelInversion_WhiteBox_test import VanillaModelInversion_WhiteBox_test

from evaluates.attacks.DLG_LLM import DLG_LLM

from evaluates.attacks.VanillaModelInversion_BlackBox import VanillaModelInversion_BlackBox
from evaluates.attacks.WhiteBoxInversion import WhiteBoxInversion
from evaluates.attacks.WhiteBoxInversion_mse import WhiteBoxInversion_mse

from evaluates.attacks.BatchLabelReconstruction_LLM import BatchLabelReconstruction_LLM # 3slice
from evaluates.attacks.BatchLabelReconstruction_LLM_2slice import BatchLabelReconstruction_LLM_2slice
from evaluates.attacks.DirectLabelScoring_LLM import DirectLabelScoring_LLM
from evaluates.attacks.DirectionbasedScoring_LLM import DirectionbasedScoring_LLM
from evaluates.attacks.NormbasedScoring_LLM import NormbasedScoring_LLM

from evaluates.attacks.ResultReconstruction import ResultReconstruction



def AttackerLoader(vfl, args):
    attacker_name = args.attack_name
    if 'ModelCompletion' in attacker_name:
        attacker_name = 'ModelCompletion'
    # if attacker_name == "DataLabelReconstruction":
    #     assert args.batch_size == 1,'DataLabelReconstruction: require batchsize=1'
    #     attacker_name == "BatchLabelReconstruction"
    return globals()[attacker_name](vfl, args)
