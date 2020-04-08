  
# -*- coding: utf-8 -*-
"""
https://github.com/NTMC-Community/MatchZoo-py/tree/master/tutorials
https://matchzoo.readthedocs.io/en/master/model_reference.html

https://github.com/NTMC-Community/MatchZoo-py/blob/master/tutorials/classification/esim.ipynb

"""
import os, json
import sys
print(sys.path)
#import torch
import matchzoo as mz
import numpy as np
import pandas as pd


from mlmodels.util import os_package_root_path, log, path_norm, get_model_uri, path_norm_dict


VERBOSE = True
MODEL_URI = get_model_uri(__file__)




#### Load all models
from mlmodels.model_keras.raw import keras_gan as kg


MODEL_MAPPING = {
    'AAE': kg
}


def get_config_file():
    return os.path.join(os_package_root_path(__file__, 1), 'config', 'model_tch', 'Imagecnn.json')


###########################################################################################################
###########################################################################################################
class Model:
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None, out_pars=None):
        ### Model Structure        ################################

        if model_pars is None :
            self.model = None
            return self
 
        _model = model_pars['model']
        assert _model in MODEL_MAPPING.keys()

        task_name = model_pars['task_name']   

        if task_name == "classification" :
            self.task = mz.tasks.Classification(num_classes= data_pars["num_classes"])
            self.task.metrics = [
                mz.metrics.accuracy(),
            ]
            self.model.params['task'] = self.task
        else :
            raise Exception(f"Not support choice {task_name} yet")


        self.model = MODEL_MAPPING[_model]()
        self.model.params['mode'] = model_pars['mode']
        self.model.params['dropout_rate'] = model_pars['dropout_rate']
        self.model.build()


def get_params(param_pars=None, **kw):
    pp          = param_pars
    choice      = pp['choice']
    config_mode = pp['config_mode']
    data_path   = pp['data_path']

    if choice == "json":
        data_path = path_norm(data_path)
        cf = json.load(open(data_path, mode='r'))
        cf = cf[config_mode]

        ####Normalize path  : add /models/dataset/
        cf['data_pars'] = path_norm_dict(cf['data_pars'])
        cf['out_pars']  = path_norm_dict(cf['out_pars'])

        return cf['model_pars'], cf['data_pars'], cf['compute_pars'], cf['out_pars']

    else:
        raise Exception(f"Not support choice {choice} yet")



def get_dataset(data_pars=None, **kw):
    data_path        = data_pars['data_path']
    train_batch_size = data_pars['train_batch_size']
    test_batch_size  = data_pars['test_batch_size']

    if data_pars['dataset'] == 'WIKI_QA':

        return train_loader, test_loader  

    else:
        raise Exception("Dataloader not implemented")
        exit



def fit(model, data_pars=None, compute_pars=None, out_pars=None, **kwargs):
    model0        = model.model
    lr            = compute_pars['learning_rate']
    epochs        = compute_pars["epochs"]
    beta1         = compute_pars["beta1"]
    beta2         = compute_pars["beta2"]
    eps           = compute_pars["eps"]

    os.makedirs(out_pars["checkpointdir"], exist_ok=True)

    return model, None


def predict(model, session=None, data_pars=None, compute_pars=None, out_pars=None):
    # get a batch of data
    model0 = model.model
    _, valid_iter = get_dataset(data_pars=data_pars)

    return ypred


def fit_metrics(model, data_pars=None, compute_pars=None, out_pars=None):
    pass


def save(model, session=None, save_pars=None):
    from mlmodels.util import save_tch
    save_tch(model=model.model, save_pars=save_pars)


def load(load_pars):
    from mlmodels.util import load_tch
    return load_tch(load_pars)



###########################################################################################################
###########################################################################################################
def test(data_path="dataset/", pars_choice="json", config_mode="test"):
    ### Local test

    log("#### Loading params   ##############################################")
    param_pars = {"choice":pars_choice,  "data_path":data_path,  "config_mode": config_mode}
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
    log(  data_pars, out_pars )

    log("#### Loading dataset   #############################################")
    #xtuple = get_dataset(data_pars)


    log("#### Model init, fit   #############################################")
    session = None
    model = Model(model_pars, data_pars, compute_pars)
    model, session = fit(model, data_pars, compute_pars, out_pars)


    log("#### Predict   #####################################################")
    #ypred = predict(model, session, data_pars, compute_pars, out_pars)


    log("#### metrics   #####################################################")
    #metrics_val = fit_metrics(model, data_pars, compute_pars, out_pars)
    #print(metrics_val)


    log("#### Plot   ########################################################")


    log("#### Save/Load   ###################################################")
    save_pars = { "path": out_pars["path"]  }
    save(model=model, save_pars=save_pars)
    model2 = load( save_pars )
    ypred = predict(model2, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
    print(model2)



if __name__ == "__main__":
    test(data_path="model_keras/keras_gan.json", pars_choice="json", config_mode="test")




