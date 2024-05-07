# copied from proof.py, but prints the stack at every step
from metamathpy.proof import *

def verify_compressed_proof_stack_trace(database, claim):

    # extract labels and mixed-radix pointer encodings
    split = claim.consequent.proof.index(")")
    step_labels = claim.consequent.proof[1:split]
    proof_string = ''.join(claim.consequent.proof[split+1:])

    # convert to integer pointers and save tagged steps
    A, U, Z = ord('A'), ord('U'), ord('Z')
    step_pointers = []
    pointer = 0
    for ordinal in map(ord, proof_string):
        if ordinal < U:
            pointer = 20 * pointer + (ordinal - A)
            step_pointers.append(pointer)
            pointer = 0
        elif ordinal < Z:
            pointer = 5 * pointer + (ordinal - U) + 1
        else:
            step_pointers.append(-1) # indicates previous step should be tagged

    # initialize proof stack
    stack = []

    # initialize buffer that gets dereferenced by step pointers
    proof_steps = []
    # steps for claim hypotheses
    for hypothesis in claim.hypotheses:
        conclusion, rule = tuple(hypothesis.tokens), database.rules[hypothesis.label]
        proof_steps.append(ProofStep(conclusion, rule))
    # step labels
    proof_steps += step_labels

    # process each step in proof
    proof_step_dict = {}
    for step, pointer in enumerate(step_pointers):

        # tag previous step if requested
        if pointer < 0: proof_steps.append(stack[-1])

        # otherwise handle current proof step
        else:

            # retrieve current step
            proof_step = proof_steps[pointer]

            # replace labels by associated step
            current_label = None
            if type(proof_step) is str:
                current_label = proof_step
                proof_step, msg = conduct(database.rules[proof_step], stack, claim.disjoint)
                assert msg == "", msg
                proof_step_dict[proof_step.conclusion] = proof_step

            # push current proof step onto stack
            stack.append(proof_step)

            # print current stack
            print(f"step {step} stack:")
            print([" ".join(s.conclusion) for s in stack])
            if current_label is not None: print(f"(rule was {current_label})")

    # check that original claim has been proved
    assert stack[0].conclusion == tuple(claim.consequent.tokens), \
           f"proved statement {' '.join(stack[0].conclusion)} does not match theorem {' '.join(claim.consequent.tokens)}"
    assert len(stack) == 1, f"non-singleton stack {stack} after proof"

    # return root of proof graph and dictionary of nodes
    return stack[0], proof_step_dict

def generate_proof_stack_trace(database, claim):

    # extract labels and mixed-radix pointer encodings
    split = claim.consequent.proof.index(")")
    step_labels = claim.consequent.proof[1:split]
    proof_string = ''.join(claim.consequent.proof[split+1:])
    stack_states = []
    # convert to integer pointers and save tagged steps
    A, U, Z = ord('A'), ord('U'), ord('Z')
    step_pointers = []
    pointer = 0
    for ordinal in map(ord, proof_string):
        if ordinal < U:
            pointer = 20 * pointer + (ordinal - A)
            step_pointers.append(pointer)
            pointer = 0
        elif ordinal < Z:
            pointer = 5 * pointer + (ordinal - U) + 1
        else:
            step_pointers.append(-1) # indicates previous step should be tagged

    # initialize proof stack
    stack = []

    # initialize buffer that gets dereferenced by step pointers
    proof_steps = []
    # steps for claim hypotheses
    for hypothesis in claim.hypotheses:
        conclusion, rule = tuple(hypothesis.tokens), database.rules[hypothesis.label]
        proof_steps.append(ProofStep(conclusion, rule))
    # step labels
    proof_steps += step_labels

    # process each step in proof
    proof_step_dict = {}
    for step, pointer in enumerate(step_pointers):

        # tag previous step if requested
        if pointer < 0: proof_steps.append(stack[-1])

        # otherwise handle current proof step
        else:

            # retrieve current step
            proof_step = proof_steps[pointer]

            # replace labels by associated step
            current_label = None
            if type(proof_step) is str:
                current_label = proof_step
                proof_step, msg = conduct(database.rules[proof_step], stack, claim.disjoint)
                assert msg == "", msg
                proof_step_dict[proof_step.conclusion] = proof_step

            # push current proof step onto stack
            stack.append(proof_step)

            # print current stack
            #print(f"step {step} stack:")
            #print([" ".join(s.conclusion) for s in stack])
            stack_states.append([" ".join(s.conclusion) for s in stack])
            #if current_label is not None: print(f"(rule was {current_label})")

    # check that original claim has been proved
    assert stack[0].conclusion == tuple(claim.consequent.tokens), \
           f"proved statement {' '.join(stack[0].conclusion)} does not match theorem {' '.join(claim.consequent.tokens)}"
    assert len(stack) == 1, f"non-singleton stack {stack} after proof"

    # return root of proof graph and dictionary of nodes
    return stack[0], proof_step_dict, stack_states


def pad_trace(stacks):
    max_len = max(map(len, stacks))
    padded = tr.full((len(stacks), max_len), PAD_IDX)
    pad_mask = tr.full(padded.shape, True)
    for s, stack in enumerate(stacks):
        padded[s,:len(stack)] = tr.tensor(stack)
        pad_mask[s,:len(stack)] = False
    return padded, pad_mask
