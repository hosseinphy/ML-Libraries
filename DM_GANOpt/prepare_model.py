from dotenv import dotenv_values
import json
import requests
import sympy as sp
import sys
from gan_manager import GANManager


def process_cmdline():
    argList = sys.argv
    argc = len(argList)

    # expected: instance_uuid
    if argc-1 != 1:
        raise Exception("run with incorrect number of arguments ({} != 1)".format(argc-1))

    return argList[1]

def read_conf_and_data():
    # read configuration from config.json
    fname = "config.json"
    with open(fname) as f:
        config = json.load(f)

    # get variable names
    invar = config.get('scalar_inp', [])
    outvar = config.get('scalar_out', [])

    # convert goal expression to a lambda function
    goal_expr = sp.parsing.sympy_parser.parse_expr( config['goal'] )
    lambda_var = tuple( sp.symbols( invar + outvar ) )
    goal_lambda = sp.lambdify(lambda_var, goal_expr)

    # get input/output values, calculate loss
    fname = "dataset.json"
    with open(fname) as f:
        dataset = json.load(f)

    inpar = []
    loss = []

    for exp in dataset.get('complete', []):
        for subset in exp.get('actuals', []):
            ss_in = [ subset['inputs'][iv] for iv in invar ]
            ss_out = [ subset['outputs'][ov] for ov in outvar ]
            par = ss_in + ss_out

            inpar.append(ss_in)
            loss.append( [ goal_lambda(*par), ] )

    return (inpar, loss)


if __name__ == "__main__":

    instance_uuid = process_cmdline()
    env_confs = dotenv_values()
    inpar, loss = read_conf_and_data()

    
    # create a batch of voxel files
    gm = GANManager()
    gm.generate_structure(instance_uuid)

    progress = {
                'min_goal_uuid': None,
                'min_goal_value': None,
                'max_goal_uuid': None,
                'max_goal_value': None,
            }

    api_base = env_confs.get('dm_executor_url')
    url_prepared = "{}/{}/prepared".format(api_base, instance_uuid)

    ## auth token required when authentication enabled on decision maker executor
    #auth_token = env_confs.get('auth_token')
    #header = { 'Authorization': auth_token }
    #requests.post( url_prepared, headers=header )
    
    print("uncomment the REQUEST line for actual run")
    #response = requests.post( url_prepared, json=progress )

    #print(url_prepared)
    #print(progress)
