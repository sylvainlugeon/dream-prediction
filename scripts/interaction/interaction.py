import yaml
import os
import shutil
from typing import Dict, Any
    
def ask_for_config(config: Dict[str, Any]) -> bool:
    
    print('\n*** CONFIG ***\n')
    print(yaml.dump(config))
    print('\n**************\n')
    
    while True:
        
        user_input = input('Continue? (y/n): ')
        
        if user_input == 'n':
            return False
        elif user_input == 'y':
            return True
        else:
            print('Answer must be "y" or "n"')
            
            
def ask_for_dir(dir: str) -> bool:
    if os.path.exists(dir):
        while True:
            delete = input(f'Are you sure you want to overwrite {dir}? (y/n): ')
            if delete == 'y':
                shutil.rmtree(dir)
                break
            elif delete == 'n':
                return False
            else:
                print('Answer must be "y" or "n"')
            
    os.makedirs(dir)
    return True