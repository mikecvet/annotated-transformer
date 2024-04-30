import math
import torch
import tqdm
from collections import Counter
from copy import deepcopy

# The following code lifted in its entirety from the following repo for convenience:
#   https://github.com/mikecvet/beam

class ScoredToken():
  """
  Represents a token and a corresponding score, which should roughly translate to the likelihood this
  token should be appended to some given generation sequence.
  """
  def __init__(self, token_id, score):
    self.token_id = token_id
    self.score = score

  def __str__(self):
    return f"{self.token_id}: {self.score: .8f}"
  
  def __repr__(self):
    return self.__str__()

class GeneratedSequence():
  """
  Represents a sequence in the process of being generated; an initial token, a potential end token, and a series of 
  ScoredTokens between them. This class also maintains the overall sequence score, which is the cumulative probability 
  of this generated sequence being the best output given some query.
  """
  def __init__(self, tokenizer, initial_token, end_token_id, initial_score):
    self.tokenizer = tokenizer
    self.end_token_id = end_token_id
    self._score = initial_score # Cumulative log probs of this sequence
    self.normalized_score = initial_score
    self.sequence = [ScoredToken(initial_token, initial_score)]
  
  def append(self, scored_token):
    """
    Append the given ScoredToken to this sequence; add its log-probability to this
    sequence's total cumulative log-prob
    """
    self.sequence.append(scored_token)
    penalty = 0.0

    if len(self.sequence) < 10:
      #penalty = -3 * math.log(10 - len(self.sequence))
      penalty = -math.pow(2, 10 - len(self.sequence))

    self._score += scored_token.score + penalty
    self.normalized_score = self._score / len(self.sequence)

  def ids(self):
    return [st.token_id for st in self.sequence]

  def tokens(self):
    return self.tokenizer.decode(torch.tensor(self.ids()), skip_special_tokens=True)
  
  def has_ended(self):
    """
    Returns True if the last token in this sequence is the end-of-sequence token ID
    """
    return self.sequence and self.sequence[-1].token_id == self.end_token_id

  def __str__(self):
    return f"{self._score: .8f}({self.normalized_score: .8f}): {self.sequence}"

  def __repr__(self):
    return self.__str__()
  
  def __copy__(self):
    gs = GeneratedSequence(self.tokenizer, None, self.end_token_id, 0.0)
    gs.sequence = self.sequence.copy()
    gs._score = self._score
    gs.normalized_score = self.normalized_score
    return gs

  def __iter__(self):
    return self.sequence.__iter__()
  
  def __lt__(self, other_sequence):
   return self.normalized_score < other_sequence.normalized_score

  def __le__(self, other_sequence):
    return self.normalized_score <= other_sequence.normalized_score

  def __eq__(self, other_sequence):
    return self.normalized_score - other_sequence.normalized_score <= 1e-5 and self.ids() == other_sequence.ids()
  
  def __ne__(self, other_sequence):
    return self.normalized_score - other_sequence.normalized_score > 1e-5 or self.ids() != other_sequence.ids()
  
  def __gt__(self, other_sequence):
    return self.normalized_score > other_sequence.normalized_score
  
  def __ge__(self, other_sequence):
    return self.normalized_score >= other_sequence.normalized_score

def search(
    tokenizer,
    model,
    src_input_ids,
    start_token_id,
    end_token_id,
    device,
    beam_width=8,
    temperature=1.0,
    max_length=64,
    decay_repeated=False):
  """
  Performs a beam search to generate heuristically-determined best top-k text sequences from a model.

  Args:
  tokenizer (Tokenizer): Tokenizer instance used for token manipulation.
  model (Model): The model used for generating predictions.
  src_input_ids (torch.Tensor): Source input ids for which output needs to be generated.
  start_token_id (int): The token id used to start the sequence generation.
  end_token_id (int): The token id that indicates the end of a sequence.
  device (torch.device): The device (CPU/GPU) where tensors should be allocated.
  beam_width (int, optional): The number of sequences to keep at each step of the beam search. Default is 8.
  temperature (float, optional): A factor used to model the confidence of the predictions. 
                                   A higher temperature results in more diversified outputs. Default is 1.0.
  max_length (int, optional): The maximum length of the sequence to be generated. 
                                   Default is 64.
  decay_repeated (bool, optional): Flag to apply decay to the score of repeated tokens to discourage repetition. Default is False.

  Returns:
  list: A list of generated sequences, where each sequence is represented as an object containing sequence ids, score, 
    and tokenized representation.

  Description:
  This function initiates a beam search algorithm to generate sequences from a given model. The search begins with a sequence 
  containing only the start_token_id. At each step, the model predicts the next token for each sequence in the current set of 
  candidate sequences. The top 'beam_width' tokens and their probabilities are used to extend the current sequences. This process 
  repeats until the sequences reach the maximum specified length or all sequences end with the end_token_id. 

  Optionally, the score for repeated tokens can be decayed to discourage repetition. 
  Finally, the scores of the sequences can be normalized by their length to prevent penalizing longer sequences.
  """
  
  print(f"beam search | k = {beam_width} bos = {start_token_id} eos = {end_token_id} temp = {temperature} beam_width = {beam_width}")

  src_input_ids = src_input_ids.unsqueeze(0).to(device)

  # The initial candidate sequence is simply the start token ID with a sequence score of 0
  candidate_sequences = [GeneratedSequence(tokenizer, start_token_id, end_token_id, 0.0)]

  # Build up output sequences until max_length tokens are reached
  for _ in tqdm.tqdm(range(max_length)):

    # Temporary list to store candidates for the next generation step
    next_step_candidates = []

    # Iterate through all candidate sequences; for each, generate the next most likely tokens
    # and add them to the next-step sequnce of candidates
    for candidate in candidate_sequences:
      if not candidate.has_ended(): # skip candidate sequences which have included the end-of-sequence token

        # Build a tensor out of the candidate IDs; add a single batch dimension
        gen_seq = torch.tensor(candidate.ids(), device=device).unsqueeze(0)

        # Predict next token
        output = model(src_input_ids, gen_seq)

        # Extract logits from output
        logits = output[:, -1, :]

        # Scale logits using temperature value
        scaled_logits = logits / temperature
        
        # Construct probability distribution against scaled logits through softmax activation function
        probs = torch.softmax(scaled_logits, dim=-1)

        # Select top k (beam_width) probabilities and IDs from the distribution
        top_probs, top_ids = probs.topk(beam_width)

        print(f"{candidate.normalized_score}: [{candidate.tokens()}], next token probabilities:")
        for p, w in zip(top_probs.tolist()[0], tokenizer.convert_ids_to_tokens(top_ids.tolist()[0])):
          print(f"\tp: {p: .8f}: {w}")
        print("\n")

        token_counts = Counter(t.token_id for t in candidate)

        # For each of the top-k generated tokens, append to this candidate sequence,
        # update its score, and append to the list of next step candidates
        for i in range(beam_width):
          next_token_id = top_ids[:, i].item() # the new token ID
          next_score = torch.log(top_probs[:, i]).item() # log-prob of the above token

          # Optionally apply a token-specific score decay to repeated tokens
          if decay_repeated and next_token_id in token_counts:
            count = token_counts[next_token_id]
            decay = 1 + math.log(count + 1)
            print(f"{tokenizer.convert_ids_to_tokens([next_token_id])} count: {count} decay: {decay}, score: {next_score}, next: {next_score * decay}")
            next_score *= decay

          new_seq = deepcopy(candidate)

          # Adds the new token to the end of this sequence, and updates its raw and normalized scores
          # Scores are normalized by sequence token length, to avoid penalizing longer sequences
          new_seq.append(ScoredToken(next_token_id, next_score))

          # Append the updated sequence to the next candidate sequence set
          next_step_candidates.append(new_seq)
      else:
        # Append the canddiate sequence as-is to the next-step candidates if it already contains an end-of-sequence token
        next_step_candidates.append(candidate)

    print(f"next step candidates:")
    for seq in reversed(sorted(next_step_candidates)):
      print(f"\t{seq.normalized_score: .8f}: [{seq.tokens()}]")
    print("\n")

    # Sort the next-step candidates by their score, select the top-k (beam_width) scoring sequences
    # and make them the new candidate_sequences list
    next_step_candidates.sort()
    candidate_sequences = list(reversed(next_step_candidates))[:beam_width]

    # Break if all sequences in the heap end with the eos_token_id
    if all(seq.has_ended() for seq in candidate_sequences):
      break

  return candidate_sequences[0].ids()
