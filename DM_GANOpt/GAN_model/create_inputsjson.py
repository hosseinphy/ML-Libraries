import json
from Params import input_dict



def jsonify_input(d):

    data = [
	{
	    'module_name': m, 
	    'inputs': [i for item in 
			   [
				[
                                    {'input_name': key, 'input_value':val} for (key, val) in zip(d[m][k]['keys'],d[m][k]['values'])
				]  for k in d[m].keys()
				    
			   ] for i in item
		      ]
	} for m in d.keys()
    ]




    json_data = json.dumps(data)

    #return json_data
    with open('./inputs_aux.json', 'w', encoding='utf8') as f:
        json.dump(data, f, indent=4)
            


def extract_stages(d):
    # return stage names
    print(*[k for k in d.keys()])


if __name__=="__main__":
    jsonify_input(input_dict)
    #extract_stages(input_dict)

