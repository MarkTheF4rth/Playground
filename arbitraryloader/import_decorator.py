from initialiser import add_procedure


def procedure(name):

    def procedure_decorator(given_procedure):
        add_procedure(name, given_procedure)

        return given_procedure

    return procedure_decorator
