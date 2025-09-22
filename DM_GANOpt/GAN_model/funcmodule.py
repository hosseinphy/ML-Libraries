import re
import json
from collections import defaultdict


#================#
def ext_uuid(pat):
    z = re.findall(r"[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}", pat)
    return z[0]if z else None
#================#


#================#
def isnumber(n):
    try:
        float(n)

    except ValueError:
        return False

    return True
#================#


#================#
def isnumberlist(a):
    # flatten the list
    a = [i for item in a for i in item] 
    c = 0
    while c < len(a):
        try:
            float(a[c])
        except ValueError:
            return False
        c += 1
    return True 
#================#




def get_modules():
    
    fname = "settings.dat"
    mod_list = []

    with open(fname) as f:

        line = f.readline()
        while line:
            if line.split()[0] == "module":
                mod_list.append(line.split()[1])
            line = f.readline()
    return mod_list
    






#================#
def read_settings(module_name):
    """
    read global & local simulation parametes into a dictionary 
    """        

    fname = "settings.dat"
    input_vars = defaultdict(lambda: " ")


    float_match = re.compile("^[-+]?[0-9]*[.][0-9]+$")
    int_match = re.compile("^[-+]?[0-9]+$")


    with open(fname) as f:

        line = f.readline()
        while line:        
            if line.split()[0] == "module" and line.split()[1] == module_name:
                break
            else:
                line = f.readline()
                continue
        

        line = f.readline()
    
        while line and line.split()[0] != "module":
            l = line.split()

            if len(l) > 1:	    		
                

                if l[1].startswith('['):
                    raw = ' '.join(l[1:])  # in case the list spans multiple tokens
                    raw = raw.strip('[] \n')
                    parts = [x.strip() for x in raw.split(',') if x.strip()]

                    try:
                        input_vars[l[0]] = [float(x) if '.' in x else int(x) for x in parts]
                    except ValueError:
                        input_vars[l[0]] = parts  # fallback to list of strings

                #if l[1].startswith('['):
                #    a = [x.strip(']|[') for x in l[1].split(',')]
                #    if isnumberlist(a):
                #        input_vars[l[0]] = [int(n) for n in 
                #                             [', '.join(
                #                                     [y for y in x.strip(']|[').split(',') if y ]
                #                                     ) for x in l[1:]
                #                             ]
                #                          ]
                #    else:
                #        input_vars[l[0]] = [x.strip(']|[') for x in l[1].split(', ')] 

                        
                elif isnumber(l[1]):
     
                    if(float_match.match(l[1])):
                        input_vars[l[0]] = float(l[1])


                    elif(int_match.match(l[1])):
                        input_vars[l[0]] = int(l[1])

                else:
                    input_vars[l[0]] = l[1]

            else:
                input_vars[l[0]] = ""

            line = f.readline()        
 
    return input_vars
#================#



def get_module_inputs():

    modes = get_modules()

    inps = defaultdict(lambda: None)

    for mode in modes:
        input_vars = read_settings(mode)        
        # extract keys & values defined as Map input values
        keys = [x.replace(mode+'_', '', 1)  for x in input_vars.keys()]
        # extract values of the input_vars
        vals = [x  for x in input_vars.values()]
        # input dictionary
        inps[mode] = dict( zip(keys, vals) )

    return inps

#================#
def read_config(module_name):
    '''
    Read outvars from "module_name"
    '''

    fname = "module_"+module_name+"_results.json"

    with open(fname, "r") as f:
        json_data = json.loads(f.read())


    return [{x['name']:x['value'] for x in json_data}]        
#================#


#================#
def sample_config(n):
    #if n == 0:
    #    return [{"internal_ref": "base", "subsamples": []}]
    #else:
    keys, vals = ['a'+ str(i) for i in range(1, n+1)], ['sample'+ str(i) for i in range(1, n+1)]
    return [{"internal_ref": "base", "subsamples": [{'internal_ref': k, 'label': v} for (k, v) in zip(keys, vals)]}]
#================#
