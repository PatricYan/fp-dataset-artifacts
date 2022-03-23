
#train version 1
task: qa 
train_data: ./squad-retrain-data/train-v1.1.json 

data 
    squad {'eval_exact_match': 78.42005676442763, 'eval_f1': 86.14486988964737}
    squad_adversarial:AddSent {'eval_exact_match': 56.17977528089887, 'eval_f1': 62.52036816376881}
    squad_adversarial:AddOneSent {'eval_exact_match': 65.5288192501399, 'eval_f1': 71.98387561817316}


#train version 2
task: qa 
train_data: ./squad-retrain-data/train-v2.0.json

data
    squad {'eval_exact_match': 77.86187322611164, 'eval_f1': 86.12484716809006}
    squad_adversarial:AddSent {'eval_exact_match': 59.80337078651685, 'eval_f1': 67.955667779176}
    squad_adversarial:AddOneSent {'eval_exact_match': 66.81589255735871, 'eval_f1': 74.93910721520925}


