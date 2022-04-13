import yaml
    
def ask_for_config(config):
    
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
