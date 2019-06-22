from math import exp

class BeamEntry:
    "information about one single beam at specific time-step"

    def __init__(self):
        #
        # probability from image
        self.pr_total = 0  # blank and non-blank
        self.pr_non_blank = 0  # non-blank
        self.pr_blank = 0  # blank

        #
        # probability from language model
        self.pr_lm_text = 1  # LM score, just equal to 1. / LogProbSent
        self.log_prob = 0. # (log probability of sentence (not including EOS token))
        self.log_prob_eos = 0. # (log probability of EOS of this sentence only)
        self.log_prob_sent = 0. # (log probability of whole sentence, including EOS)
        self.lm_applied = False  # flag if LM was already applied to this beam

        #
        # string labeling
        self.labeling = ()  # beam-labeling
        self.labeling_str = ""

class BeamContainer:
    "information about the beams at specific time-step"

    def __init__(self):
        self.entries = {} # list of beam entries

    def sort(self):
        "return beam-labelings, sorted by probability (including language model here)"
        beams = [v for (_, v) in self.entries.items()]

        sortedBeams = sorted(beams, reverse=True, key=lambda x: x.pr_total * x.pr_lm_text)
        return [x.labeling for x in sortedBeams]

    def add_beam(self, labeling):
        if labeling not in self.entries:
            self.entries[labeling] = BeamEntry()

    @staticmethod
    def initialize():
        beam_container = BeamContainer()

        labeling = ()
        labeling_str = ""

        beam_container.entries[labeling] = BeamEntry()
        beam_container.entries[labeling].pr_blank = 1
        beam_container.entries[labeling].pr_total = 1

        return beam_container

class BeamSearch:
    def __init__(self, beam_width, lm_factor, topk, classes, lm_model=None):
        self.beam_width = beam_width
        self.lm_factor = lm_factor
        self.topk = topk
        self.classes = classes
        self.blank_idx = -1

        self.lm_model = lm_model

    def update_lm_for_new_beam(self, parent_beam, child_beam):
        if self.lm_model and not child_beam.lm_applied:
            child_str  = child_beam.labeling_str
            parent_str = parent_beam.labeling_str
            new_c = child_str[-1]

            # conditional probability
            parent2child_log_prob = self.lm_model.get_log_prob(parent_str, new_c)
            child_eos_log_prob = self.lm_model.get_log_prob_of_EOS(child_str)

            # fill in language model for beam entry
            child_beam.log_prob = parent_beam.log_prob + parent2child_log_prob
            child_beam.log_prob_eos = child_eos_log_prob
            child_beam.log_prob_sent = child_beam.log_prob + child_beam.log_prob_eos
            child_beam.pr_lm_text = exp(child_beam.log_prob_sent / (len(child_str) + 1)) ** self.lm_factor

            # mask
            child_beam.lm_applied = True

    def search(self, ctc_logit_matrix):
        max_t, max_c = ctc_logit_matrix.shape

        last = BeamContainer.initialize()

        # go over all time-steps
        for t in range(max_t):
            curr = BeamContainer()

            # get beam labelings of best beams
            for labeling in last.sort()[0:self.beam_width]:
                # probability of paths ending with a non-blank
                pr_non_blank = 0

                ######################
                ##### COPY PHASE #####
                ######################
                # in case of non-empty beam
                if labeling:
                    # probability of paths with repeated last char at the end
                    pr_non_blank = last.entries[labeling].pr_non_blank * ctc_logit_matrix[t, labeling[-1]]

                # probability of paths ending with a blank
                pr_blank = (last.entries[labeling].pr_total) * ctc_logit_matrix[t, self.blank_idx]

                # add beam at current time-step if needed
                curr.add_beam(labeling)

                # fill in data (score from ctc matrix score)
                curr.entries[labeling].pr_non_blank += pr_non_blank
                curr.entries[labeling].pr_blank += pr_blank
                curr.entries[labeling].pr_total += pr_blank + pr_non_blank
                curr.entries[labeling].pr_lm_text = last.entries[
                    labeling].pr_lm_text  # beam-labeling not changed, therefore also LM score unchanged from
                curr.entries[
                    labeling].lm_applied = True  # LM already applied at previous time-step for this beam-labeling

                # fill in language model score
                curr.entries[labeling].log_prob = last.entries[labeling].log_prob
                curr.entries[labeling].log_prob_eos = last.entries[labeling].log_prob_eos
                curr.entries[labeling].log_prob_sent = last.entries[labeling].log_prob_sent

                # fill in label
                curr.entries[labeling].labeling = labeling
                curr.entries[labeling].labeling_str = last.entries[labeling].labeling_str

                #########################
                ###### EXTEND PHASE #####
                #########################
                # consider only the top-k highest characters
                considered_cs = ctc_logit_matrix[t, :-1].argsort(kind='quicksort')[-self.topk:]
                for c in considered_cs:
                    # add new char to current beam-labeling
                    new_labeling = labeling + (c,)

                    # if new labeling contains duplicate char at the end, only consider paths ending with a blank
                    if labeling and labeling[-1] == c:
                        pr_non_blank = ctc_logit_matrix[t, c] * last.entries[labeling].pr_blank
                    else:
                        pr_non_blank = ctc_logit_matrix[t, c] * last.entries[labeling].pr_total

                    # add beam at current time-step if needed
                    curr.add_beam(new_labeling)

                    # fill in data (score from ctc matrix score)
                    curr.entries[new_labeling].pr_non_blank += pr_non_blank
                    curr.entries[new_labeling].pr_total += pr_non_blank

                    # fill in label
                    curr.entries[new_labeling].labeling_str = last.entries[labeling].labeling_str + self.classes[c]
                    curr.entries[new_labeling].labeling = new_labeling

                    # fill in language model score
                    self.update_lm_for_new_beam(curr.entries[labeling], curr.entries[new_labeling])

            # here
            # set new beam container
            last = curr

        # sort by probability (both from ctc matrix score & language model score)
        best_labeling_ever = last.sort()[0]
        return "".join([self.classes[i] for i in best_labeling_ever])