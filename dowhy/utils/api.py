def parse_state(state):
    if type(state) == str:
        return [state]
    if type(state) == list:
        return state
    if type(state) == dict:
        return list(state.keys())
    if not state:
        return []
    raise Exception(f'Input format for {state} not recognized: {type(state)}')
