import initialiser

if __name__ == "__main__":
    procedures = initialiser.initialise()
    for name, function in procedures.items():
        output = function()
        print(f'Procedure {name} has output {output}')
