from SRC.Preprocessing.Botnet_Detection_Model_Prediction_And_Interference_Pipeline_Processing import Botnet_Detection_Model_Prediction_And_Interference_Pipeline_Processing as BotnetP
from SRC.Preprocessing.Botnet_Detection_Model_Preprocessing import Botnet_Detection_Model_Preprocessing as Botnet
from SRC.Preprocessing.DDoS_Model_Prediction_And_Inference_Pipeline_Processing import DDoS_Model_Prediction_And_Inference_Pipeline_Processing as DDoSP
from SRC.Preprocessing.DDoS_Model_Preprocessing import DDoS_Model_Preprocessing as DDoS
from SRC.Preprocessing.Network_Anomaly_Model_Predicition_And_Interference_Pipeline_Processing import Network_Anomaly_Model_Predicition_And_Interference_Pipeline_Processing as NtwrkP
from SRC.Preprocessing.Network_Anomaly_Preprocessing import Network_Anomaly_Preprocessing as Ntwrk
from SRC.Preprocessing.NIDS_Model_Prediction_And_Interference_Pipieline_Processing import NIDS_Model_Prediction_And_Interference_Pipieline_Processing as NIDSP
from SRC.Preprocessing.NIDS_Model_Preprocessing import NIDS_Model_Preprocessing as NIDS
from SRC.Training.Botnet_Detection_Model_Traning import Botnet_Detection_Model_Traning as BotnetM
from SRC.Training.DDoS_Model_Traning import DDoS_Model_Traning as DDoSM
from SRC.Training.Network_Anomaly_Model_Traning import Network_Anomaly_Model_Traning as NtwrkM
from SRC.Training.NIDS_Model_Traning import NIDS_Model_Traning as NIDSM

#Here all the CSVs are Processed
Botnet()
DDoS()
Ntwrk()
NIDS()
BotnetP()
DDoSP()
NtwrkP()
NIDSP()

#Here all the Models are tranied and made and the prints of all the stats
print("\n")
print("These are the scores of the Botnet Model")
BotnetM()
print("\n\nThese are the scores of the DDoS Model")
DDoSM()
print("\n\nThese are the scores of the Network Anomaly Model")
NtwrkM()
print("\n\nThese are the scores of the NIDS Model")
NIDSM()
print("\n")