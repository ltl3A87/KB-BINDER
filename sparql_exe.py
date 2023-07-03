from typing import List, Tuple
from SPARQLWrapper import SPARQLWrapper, JSON
import json
import urllib
from pathlib import Path
from tqdm import tqdm

sparql = SPARQLWrapper("http://129.97.152.19:3001/sparql")
sparql.setReturnFormat(JSON)

path = str(Path(__file__).parent.absolute())

with open(path + '/GrailQA/ontology/fb_roles', 'r') as f:
    contents = f.readlines()

roles = set()
for line in contents:
    fields = line.split()
    roles.add(fields[1])


def lisp_to_nested_expression(lisp_string: str) -> List:
    """
    Takes a logical form as a lisp string and returns a nested list representation of the lisp.
    For example, "(count (division first))" would get mapped to ['count', ['division', 'first']].
    """
    stack: List = []
    current_expression: List = []
    tokens = lisp_string.split()
    for token in tokens:
        while token[0] == '(':
            nested_expression: List = []
            current_expression.append(nested_expression)
            stack.append(current_expression)
            current_expression = nested_expression
            token = token[1:]
        current_expression.append(token.replace(')', ''))
        while token[-1] == ')':
            current_expression = stack.pop()
            token = token[:-1]
    return current_expression[0]

def _linearize_lisp_expression(expression: list, sub_formula_id):
    sub_formulas = []
    for i, e in enumerate(expression):
        if isinstance(e, list) and e[0] != 'R':
            sub_formulas.extend(_linearize_lisp_expression(e, sub_formula_id))
            expression[i] = '#' + str(sub_formula_id[0] - 1)

    sub_formulas.append(expression)
    sub_formula_id[0] += 1
    return sub_formulas


def lisp_to_sparql(lisp_program: str):
    clauses = []
    order_clauses = []
    entities = set()  # collect entites for filtering
    # identical_variables = {}   # key should be smaller than value, we will use small variable to replace large variable
    identical_variables_r = {}  # key should be larger than value
    expression = lisp_to_nested_expression(lisp_program)
    superlative = False
    if expression[0] in ['ARGMAX', 'ARGMIN']:
        superlative = True
        # remove all joins in relation chain of an arg function. In another word, we will not use arg function as
        # binary function here, instead, the arity depends on the number of relations in the second argument in the
        # original function
        if isinstance(expression[2], list):
            def retrieve_relations(exp: list):
                rtn = []
                for element in exp:
                    if element == 'JOIN':
                        continue
                    elif isinstance(element, str):
                        rtn.append(element)
                    elif isinstance(element, list) and element[0] == 'R':
                        rtn.append(element)
                    elif isinstance(element, list) and element[0] == 'JOIN':
                        rtn.extend(retrieve_relations(element))
                return rtn

            relations = retrieve_relations(expression[2])
            expression = expression[:2]
            expression.extend(relations)

    sub_programs = _linearize_lisp_expression(expression, [0])
    question_var = len(sub_programs) - 1
    count = False

    def get_root(var: int):
        while var in identical_variables_r:
            var = identical_variables_r[var]

        return var

    for i, subp in enumerate(sub_programs):
        i = str(i)
        if subp[0] == 'JOIN':
            if isinstance(subp[1], list):  # R relation
                if subp[2][:2] in ["m.", "g."]:  # entity
                    clauses.append("ns:" + subp[2] + " ns:" + subp[1][1] + " ?x" + i + " .")
                    entities.add(subp[2])
                elif subp[2][0] == '#':  # variable
                    clauses.append("?x" + subp[2][1:] + " ns:" + subp[1][1] + " ?x" + i + " .")
                else:  # literal   (actually I think literal can only be object)
                    if subp[2].__contains__('^^'):
                        data_type = subp[2].split("^^")[1].split("#")[1]
                        if data_type not in ['integer', 'float', 'dateTime']:
                            subp[2] = f'"{subp[2].split("^^")[0] + "-08:00"}"^^<{subp[2].split("^^")[1]}>'
                            # subp[2] = subp[2].split("^^")[0] + '-08:00^^' + subp[2].split("^^")[1]
                        else:
                            subp[2] = f'"{subp[2].split("^^")[0]}"^^<{subp[2].split("^^")[1]}>'
                    clauses.append(subp[2] + " ns:" + subp[1][1] + " ?x" + i + " .")
            else:
                if subp[2][:2] in ["m.", "g."]:  # entity
                    clauses.append("?x" + i + " ns:" + subp[1] + " ns:" + subp[2] + " .")
                    entities.add(subp[2])
                elif subp[2][0] == '#':  # variable
                    clauses.append("?x" + i + " ns:" + subp[1] + " ?x" + subp[2][1:] + " .")
                else:  # literal
                    if subp[2].__contains__('^^'):
                        data_type = subp[2].split("^^")[1].split("#")[1]
                        if data_type not in ['integer', 'float', 'dateTime']:
                            subp[2] = f'"{subp[2].split("^^")[0] + "-08:00"}"^^<{subp[2].split("^^")[1]}>'
                        else:
                            subp[2] = f'"{subp[2].split("^^")[0]}"^^<{subp[2].split("^^")[1]}>'
                    clauses.append("?x" + i + " ns:" + subp[1] + " " + subp[2] + " .")
        elif subp[0] == 'AND':
            var1 = int(subp[2][1:])
            rooti = get_root(int(i))
            root1 = get_root(var1)
            if rooti > root1:
                identical_variables_r[rooti] = root1
            else:
                identical_variables_r[root1] = rooti
                root1 = rooti
            # identical_variables[var1] = int(i)
            if subp[1][0] == "#":
                var2 = int(subp[1][1:])
                root2 = get_root(var2)
                # identical_variables[var2] = int(i)
                if root1 > root2:
                    # identical_variables[var2] = var1
                    identical_variables_r[root1] = root2
                else:
                    # identical_variables[var1] = var2
                    identical_variables_r[root2] = root1
            else:  # 2nd argument is a class
                clauses.append("?x" + i + " ns:type.object.type ns:" + subp[1] + " .")
        elif subp[0] in ['le', 'lt', 'ge', 'gt']:  # the 2nd can only be numerical value
            clauses.append("?x" + i + " ns:" + subp[1] + " ?y" + i + " .")
            if subp[0] == 'le':
                op = "<="
            elif subp[0] == 'lt':
                op = "<"
            elif subp[0] == 'ge':
                op = ">="
            else:
                op = ">"
            if subp[2].__contains__('^^'):
                data_type = subp[2].split("^^")[1].split("#")[1]
                if data_type not in ['integer', 'float', 'dateTime']:
                    subp[2] = f'"{subp[2].split("^^")[0] + "-08:00"}"^^<{subp[2].split("^^")[1]}>'
                else:
                    subp[2] = f'"{subp[2].split("^^")[0]}"^^<{subp[2].split("^^")[1]}>'
            clauses.append(f"FILTER (?y{i} {op} {subp[2]})")
        elif subp[0] == 'TC':
            var = int(subp[1][1:])
            # identical_variables[var] = int(i)
            rooti = get_root(int(i))
            root_var = get_root(var)
            if rooti > root_var:
                identical_variables_r[rooti] = root_var
            else:
                identical_variables_r[root_var] = rooti

            year = subp[3]
            if year == 'NOW':
                from_para = '"2015-08-10"^^xsd:dateTime'
                to_para = '"2015-08-10"^^xsd:dateTime'
            else:
                from_para = f'"{year}-12-31"^^xsd:dateTime'
                to_para = f'"{year}-01-01"^^xsd:dateTime'

            clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2]} ?sk0}} || ')
            clauses.append(f'EXISTS {{?x{i} ns:{subp[2]} ?sk1 . ')
            clauses.append(f'FILTER(xsd:datetime(?sk1) <= {from_para}) }})')
            if subp[2][-4:] == "from":
                clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2][:-4] + "to"} ?sk2}} || ')
                clauses.append(f'EXISTS {{?x{i} ns:{subp[2][:-4] + "to"} ?sk3 . ')
            else:  # from_date -> to_date
                clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2][:-9] + "to_date"} ?sk2}} || ')
                clauses.append(f'EXISTS {{?x{i} ns:{subp[2][:-9] + "to_date"} ?sk3 . ')
            clauses.append(f'FILTER(xsd:datetime(?sk3) >= {to_para}) }})')

        elif subp[0] in ["ARGMIN", "ARGMAX"]:
            superlative = True
            if subp[1][0] == '#':
                var = int(subp[1][1:])
                rooti = get_root(int(i))
                root_var = get_root(var)
                # identical_variables[var] = int(i)
                if rooti > root_var:
                    identical_variables_r[rooti] = root_var
                else:
                    identical_variables_r[root_var] = rooti
            else:  # arg1 is class
                clauses.append(f'?x{i} ns:type.object.type ns:{subp[1]} .')

            if len(subp) == 3:
                clauses.append(f'?x{i} ns:{subp[2]} ?sk0 .')
            elif len(subp) > 3:
                for j, relation in enumerate(subp[2:-1]):
                    if j == 0:
                        var0 = f'x{i}'
                    else:
                        var0 = f'c{j - 1}'
                    var1 = f'c{j}'
                    if isinstance(relation, list) and relation[0] == 'R':
                        clauses.append(f'?{var1} ns:{relation[1]} ?{var0} .')
                    else:
                        clauses.append(f'?{var0} ns:{relation} ?{var1} .')

                clauses.append(f'?c{j} ns:{subp[-1]} ?sk0 .')

            if subp[0] == 'ARGMIN':
                order_clauses.append("ORDER BY ?sk0")
            elif subp[0] == 'ARGMAX':
                order_clauses.append("ORDER BY DESC(?sk0)")
            order_clauses.append("LIMIT 1")


        elif subp[0] == 'COUNT':  # this is easy, since it can only be applied to the quesiton node
            var = int(subp[1][1:])
            root_var = get_root(var)
            identical_variables_r[int(i)] = root_var  # COUNT can only be the outtermost
            count = True
    #  Merge identical variables
    for i in range(len(clauses)):
        for k in identical_variables_r:
            clauses[i] = clauses[i].replace(f'?x{k} ', f'?x{get_root(k)} ')

    question_var = get_root(question_var)

    for i in range(len(clauses)):
        clauses[i] = clauses[i].replace(f'?x{question_var} ', f'?x ')

    if superlative:
        arg_clauses = clauses[:]

    for entity in entities:
        clauses.append(f'FILTER (?x != ns:{entity})')
    clauses.insert(0,
                   f"FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))")
    clauses.insert(0, "WHERE {")
    if count:
        clauses.insert(0, f"SELECT COUNT DISTINCT ?x")
    elif superlative:
        clauses.insert(0, "{SELECT ?sk0")
        clauses = arg_clauses + clauses
        clauses.insert(0, "WHERE {")
        clauses.insert(0, f"SELECT DISTINCT ?x")
    else:
        clauses.insert(0, f"SELECT DISTINCT ?x")
    clauses.insert(0, "PREFIX ns: <http://rdf.freebase.com/ns/>")

    clauses.append('}')
    clauses.extend(order_clauses)
    if superlative:
        clauses.append('}')
        clauses.append('}')

    # for clause in clauses:
    #     print(clause)

    return '\n'.join(clauses)



def execute_query(query: str) -> List[str]:
    # print("http://129.97.152.19:3001/sparql")
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    rtn = []
    for result in results['results']['bindings']:
        assert len(result) == 1  # only select one variable
        for var in result:
            rtn.append(result[var]['value'].replace('http://rdf.freebase.com/ns/', '').replace("-08:00", ''))

    return rtn


def execute_unary(type: str) -> List[str]:
    query = ("""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX : <http://rdf.freebase.com/ns/> 
    SELECT (?x0 AS ?value) WHERE {
    SELECT DISTINCT ?x0  WHERE {
    """
             '?x0 :type.object.type :' + type + '. '
                                                """
    }
    }
    """)
    # # print(query)
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    rtn = []
    for result in results['results']['bindings']:
        rtn.append(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return rtn


def execute_binary(relation: str) -> List[Tuple[str, str]]:
    query = ("""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX : <http://rdf.freebase.com/ns/> 
    SELECT DISTINCT ?x0 ?x1 WHERE {
    """
             '?x0 :' + relation + ' ?x1. '
                                  """
    }
    """)
    # # print(query)
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    rtn = []
    for result in results['results']['bindings']:
        rtn.append((result['x0']['value'], result['x1']['value']))

    return rtn


def get_types(entity: str) -> List[str]:
    query = ("""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX : <http://rdf.freebase.com/ns/> 
    SELECT (?x0 AS ?value) WHERE {
    SELECT DISTINCT ?x0  WHERE {
    """
             ':' + entity + ' :type.object.type ?x0 . '
                            """
    }
    }
    """)
    # print(query)
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    rtn = []
    for result in results['results']['bindings']:
        rtn.append(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return rtn


def get_notable_type(entity: str):
    query = ("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x0 AS ?value) WHERE {
        SELECT DISTINCT ?x0  WHERE {

        """
             ':' + entity + ' :common.topic.notable_types ?y . '
                            """
        ?y :type.object.name ?x0
        FILTER (lang(?x0) = 'en')
    }
    }
    """)

    # print(query)
    sparql.setQuery(query)
    results = sparql.query().convert()
    rtn = []
    for result in results['results']['bindings']:
        rtn.append(result['value']['value'])

    if len(rtn) == 0:
        rtn = ['entity']

    return rtn


def get_friendly_name(entity: str) -> str:
    query = ("""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX : <http://rdf.freebase.com/ns/> 
    SELECT (?x0 AS ?value) WHERE {
    SELECT DISTINCT ?x0  WHERE {
    """
             ':' + entity + ' :type.object.name ?x0 . '
                            """
    }
    }
    """)
    # # print(query)
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    rtn = []
    for result in results['results']['bindings']:
        if result['value']['xml:lang'] == 'en':
            rtn.append(result['value']['value'])

    if len(rtn) == 0:
        query = ("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x0 AS ?value) WHERE {
        SELECT DISTINCT ?x0  WHERE {
        """
                 ':' + entity + ' :common.topic.alias ?x0 . '
                                """
        }
        }
        """)
        # # print(query)
        sparql.setQuery(query)
        try:
            results = sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)
        for result in results['results']['bindings']:
            if result['value']['xml:lang'] == 'en':
                rtn.append(result['value']['value'])

    if len(rtn) == 0:
        return 'null'

    return rtn[0]


def get_degree(entity: str):
    degree = 0

    query1 = ("""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX : <http://rdf.freebase.com/ns/> 
            SELECT count(?x0) as ?value WHERE {
            """
              '?x1 ?x0 ' + ':' + entity + '. '
                                          """
     FILTER regex(?x0, "http://rdf.freebase.com/ns/")
     }
     """)
    sparql.setQuery(query1)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query1)
        exit(0)
    for result in results['results']['bindings']:
        degree += int(result['value']['value'])

    query2 = ("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT count(?x0) as ?value WHERE {
        """
              ':' + entity + ' ?x0 ?x1 . '
                             """
    FILTER regex(?x0, "http://rdf.freebase.com/ns/")
    }
    """)

    sparql.setQuery(query2)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query2)
        exit(0)
    for result in results['results']['bindings']:
        degree += int(result['value']['value'])

    return degree


def get_in_attributes(value: str):
    in_attributes = set()

    query1 = ("""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/> 
                SELECT (?x0 AS ?value) WHERE {
                SELECT DISTINCT ?x0  WHERE {
                """
              '?x1 ?x0 ' + value + '. '
                                   """
    FILTER regex(?x0, "http://rdf.freebase.com/ns/")
    }
    }
    """)
    # print(query1)

    sparql.setQuery(query1)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query1)
        exit(0)
    for result in results['results']['bindings']:
        in_attributes.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return in_attributes


def get_in_relations(entity: str):
    in_relations = set()

    query1 = ("""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX : <http://rdf.freebase.com/ns/> 
            SELECT (?x0 AS ?value) WHERE {
            SELECT DISTINCT ?x0  WHERE {
            """
              '?x1 ?x0 ' + ':' + entity + '. '
                                          """
     FILTER regex(?x0, "http://rdf.freebase.com/ns/")
     }
     }
     """)
    # print(query1)

    sparql.setQuery(query1)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query1)
        exit(0)
    for result in results['results']['bindings']:
        in_relations.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return in_relations


def get_in_entities(entity: str, relation: str):
    neighbors = set()

    query1 = ("""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX : <http://rdf.freebase.com/ns/> 
            SELECT (?x1 AS ?value) WHERE {
            SELECT DISTINCT ?x1  WHERE {
            """
              '?x1' + ':' + relation + ':' + entity + '. '
                                                      """
                 FILTER regex(?x1, "http://rdf.freebase.com/ns/")
                 }
                 }
                 """)
    # print(query1)

    sparql.setQuery(query1)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query1)
        exit(0)
    for result in results['results']['bindings']:
        neighbors.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return neighbors


def get_in_entities_for_literal(value: str, relation: str):
    neighbors = set()

    query1 = ("""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX : <http://rdf.freebase.com/ns/> 
            SELECT (?x1 AS ?value) WHERE {
            SELECT DISTINCT ?x1  WHERE {
            """
              '?x1' + ':' + relation + ' ' + value + '. '
                                                     """
                FILTER regex(?x1, "http://rdf.freebase.com/ns/")
                }
                }
                """)
    # print(query1)

    sparql.setQuery(query1)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query1)
        exit(0)
    for result in results['results']['bindings']:
        neighbors.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return neighbors


def get_out_relations(entity: str):
    out_relations = set()

    query2 = ("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x0 AS ?value) WHERE {
        SELECT DISTINCT ?x0  WHERE {
        """
              ':' + entity + ' ?x0 ?x1 . '
                             """
    FILTER regex(?x0, "http://rdf.freebase.com/ns/")
    }
    }
    """)
    # print(query2)

    sparql.setQuery(query2)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query2)
        exit(0)
    for result in results['results']['bindings']:
        out_relations.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return out_relations


def get_out_entities(entity: str, relation: str):
    neighbors = set()

    query2 = ("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x1 AS ?value) WHERE {
        SELECT DISTINCT ?x1  WHERE {
        """
              ':' + entity + ':' + relation + ' ?x1 . '
                                              """
                     FILTER regex(?x1, "http://rdf.freebase.com/ns/")
                     }
                     }
                     """)
    # print(query2)

    sparql.setQuery(query2)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query2)
        exit(0)
    for result in results['results']['bindings']:
        neighbors.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return neighbors


def get_entities_cmp(value, relation: str, cmp: str):
    neighbors = set()

    query2 = ("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x1 AS ?value) WHERE {
        SELECT DISTINCT ?x1  WHERE {
        """
              '?x1' + ':' + relation + ' ?sk0 . '
                                       """
              FILTER regex(?x1, "http://rdf.freebase.com/ns/")
              """
                                       f'FILTER (?sk0 {cmp} {value})'
                                       """
                                       }
                                       }
                                       """)
    # print(query2)

    sparql.setQuery(query2)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query2)
        exit(0)
    for result in results['results']['bindings']:
        neighbors.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return neighbors


def get_adjacent_relations(entity: str):
    in_relations = set()
    out_relations = set()

    query1 = ("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x0 AS ?value) WHERE {
        SELECT DISTINCT ?x0  WHERE {
        """
              '?x1 ?x0 ' + ':' + entity + '. '
                                          """
     FILTER regex(?x0, "http://rdf.freebase.com/ns/")
     }
     }
     """)

    sparql.setQuery(query1)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query1)
        exit(0)
    for result in results['results']['bindings']:
        in_relations.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    query2 = ("""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX : <http://rdf.freebase.com/ns/> 
    SELECT (?x0 AS ?value) WHERE {
    SELECT DISTINCT ?x0  WHERE {
    """
              ':' + entity + ' ?x0 ?x1 . '
                             """
    FILTER regex(?x0, "http://rdf.freebase.com/ns/")
    }
    }
    """)

    sparql.setQuery(query2)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query2)
        exit(0)
    for result in results['results']['bindings']:
        out_relations.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return in_relations, out_relations


def get_2hop_relations_from_2entities(entity0: str, entity1: str):  # m.027lnzs  m.0zd6  3200017000000
    query = ("""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX : <http://rdf.freebase.com/ns/>
            SELECT distinct ?x0 as ?r0 ?y as ?r1 WHERE {
            """
             '?x1 ?x0 ' + ':' + entity0 + ' .\n' + '?x1 ?y ' + ':' + entity1 + ' .'
                                                                               """
                                                       FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                                                       FILTER regex(?y, "http://rdf.freebase.com/ns/")
                                                       }
                                                       """)
    # print(query)
    pass


def get_2hop_relations(entity: str):
    in_relations = set()
    out_relations = set()
    paths = []

    query1 = ("""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX : <http://rdf.freebase.com/ns/>
            SELECT distinct ?x0 as ?r0 ?y as ?r1 WHERE {
            """
              '?x1 ?x0 ' + ':' + entity + '. '
                                          """
                ?x2 ?y ?x1 .
                  FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                  FILTER regex(?y, "http://rdf.freebase.com/ns/")
                  }
                  """)

    sparql.setQuery(query1)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query1)
        exit(0)
    for result in results['results']['bindings']:
        r1 = result['r1']['value'].replace('http://rdf.freebase.com/ns/', '')
        r0 = result['r0']['value'].replace('http://rdf.freebase.com/ns/', '')
        in_relations.add(r0)
        in_relations.add(r1)

        if r0 in roles and r1 in roles:
            paths.append((r0, r1))

    query2 = ("""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX : <http://rdf.freebase.com/ns/> 
            SELECT distinct ?x0 as ?r0 ?y as ?r1 WHERE {
            """
              '?x1 ?x0 ' + ':' + entity + '. '
                                          """
                ?x1 ?y ?x2 .
                  FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                  FILTER regex(?y, "http://rdf.freebase.com/ns/")
                  }
                  """)

    sparql.setQuery(query2)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query2)
        exit(0)
    for result in results['results']['bindings']:
        r1 = result['r1']['value'].replace('http://rdf.freebase.com/ns/', '')
        r0 = result['r0']['value'].replace('http://rdf.freebase.com/ns/', '')
        out_relations.add(r1)
        in_relations.add(r0)

        if r0 in roles and r1 in roles:
            paths.append((r0, r1 + '#R'))

    query3 = ("""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/>
                SELECT distinct ?x0 as ?r0 ?y as ?r1 WHERE {
                """
              ':' + entity + ' ?x0 ?x1 . '
                             """
                ?x2 ?y ?x1 .
                  FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                  FILTER regex(?y, "http://rdf.freebase.com/ns/")
                  }
                  """)

    sparql.setQuery(query3)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query3)
        exit(0)
    for result in results['results']['bindings']:
        r1 = result['r1']['value'].replace('http://rdf.freebase.com/ns/', '')
        r0 = result['r0']['value'].replace('http://rdf.freebase.com/ns/', '')
        in_relations.add(r1)
        out_relations.add(r0)

        if r0 in roles and r1 in roles:
            paths.append((r0 + '#R', r1))

    query4 = ("""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/>
                SELECT distinct ?x0 as ?r0 ?y as ?r1 WHERE {
                """
              ':' + entity + ' ?x0 ?x1 . '
                             """
                ?x1 ?y ?x2 .
                  FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                  FILTER regex(?y, "http://rdf.freebase.com/ns/")
                  }
                  """)

    sparql.setQuery(query4)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query4)
        exit(0)
    for result in results['results']['bindings']:
        r1 = result['r1']['value'].replace('http://rdf.freebase.com/ns/', '')
        r0 = result['r0']['value'].replace('http://rdf.freebase.com/ns/', '')
        out_relations.add(r1)
        out_relations.add(r0)

        if r0 in roles and r1 in roles:
            paths.append((r0 + '#R', r1 + '#R'))

    return in_relations, out_relations, paths


def get_label(entity: str) -> str:
    query = ("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x0 AS ?label) WHERE {
        SELECT DISTINCT ?x0  WHERE {
        """
             ':' + entity + ' rdfs:label ?x0 . '
                            """
                            FILTER (langMatches( lang(?x0), "EN" ) )
                             }
                             }
                             """)
    # # print(query)
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    rtn = []
    for result in results['results']['bindings']:
        label = result['label']['value']
        rtn.append(label)
    if len(rtn) != 0:
        return rtn[0]
    else:
        return None