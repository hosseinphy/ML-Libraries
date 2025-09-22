#import pandas as pd
import pandas as pd
import data_emulator as emu
import sys




def isnumber(n):
    try:
        float(n)

    except ValueError:
        return False

    return True



def read_settings():
    import re        

    fname = "settings.dat"
    input_vars = {}


    float_match = re.compile("^[-+]?[0-9]*[.][0-9]+$")
    int_match = re.compile("^[-+]?[0-9]+$")


    with open(fname) as f:
        for line in f:
            if line.startswith("#"):
                continue

            l = line.split()
            if len(l) > 1:	    			
                if l[1].startswith('['):
                    input_vars[l[0]] = [int(n) for n in 
                                            [', '.join(
                                                    [y for y in x.strip(']|[').split(',') if y ]
                                                    ) for x in l[1:]
                                            ]
                                       ]
                elif isnumber(l[1]):
    
                        if(float_match.match(l[1])):
                            input_vars[l[0]] = float(l[1])

                        elif(int_match.match(l[1])):
                            input_vars[l[0]] = int(l[1])

                else:
                        input_vars[l[0]] = l[1]
            else:
                input_vars[l[0]] = ""

    return input_vars






def calc_fun(input_vars, stage):



    fname = "stage_{}_results.csv".format(stage)


    prop = emu.PropertyEmulator(
                                    T_label=input_vars["x_label"],
                                    T_unit=input_vars['x_unit'],
                                    y_label=input_vars['y_label'], 
                                    y_unit=input_vars['y_unit']
                                )


    prop.add_T_var_cycle(
                                    intervals=input_vars['intervals'],
                                    n_pts=input_vars['n_pts'],
                                    std=input_vars['x_std']
                        )

    prop.add_y_var_cycle(
                                    y_init=input_vars['y_init'],
                                    y_offset=input_vars['y_offset'],
                                    incr_slopes=input_vars['incr_slopes'],
                                    decr_slopes=input_vars['decr_slopes'],
                                    incr_trans_pts=input_vars['incr_trans_pts'],
                                    decr_trans_pts=input_vars['decr_trans_pts'],
                                    std=input_vars['y_std']
                        )
    
    prop.save_to_csv(fname)

    df = pd.DataFrame(prop._data_dict)

    colx, coly = input_vars['x_label'], input_vars['y_label']


    #return [stage] + [
    #                    [
    #                        {
    #                            'name': '{0}{1}'.format(colx, coly), 
    #                            'value':[[x, y] for x,y in zip(df[colx].to_list(), df[coly].to_list())]
    #                        }
    #                    ]
    #                ]


    return [
		{
			'name': '{0}{1}'.format(colx, coly), 
			'value':[[x, y] for x,y in zip(df[colx].to_list(), df[coly].to_list())]
		}
	   ]
	  
	  

    #return [ {'label': k  , 'units': input_vars[k], 'values': df[k].to_list()} for k in ["Temperature", "Property"] ]
    #return  [stage] + [[{'name': k  ,'value': df[k].to_list()} for k in [input_vars['x_label'], input_vars['y_label']] ]]
    #return  [stage] + [[{'name': '{}{}', 'value': [[x, y] for (x, y) in df[k]} for k in [input_vars['x_label'], input_vars['y_label']] ]]




def write_results(output_vars, stage):

    import json

    fname = "stage_{}_results.json".format(stage)

    with open(fname, "w") as f:
        json.dump(obj=output_vars, fp=f, indent=4)

########################################

if __name__ == "__main__":
    stage_name = sys.argv[1] 
    input_vars = read_settings()
    output_vars = calc_fun(input_vars, stage_name)
    write_results(output_vars, stage_name)
