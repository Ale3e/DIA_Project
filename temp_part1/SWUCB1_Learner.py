from UCB1_Learner import *

class SWUCB1_Learner(UCB1_Learner):
    def __init__(self, n_arms, window_size, prices):
        super().__init__(n_arms, prices)
        self.window_size = window_size
        self.selections_windowed = [0.0] * n_arms



    def pull_arm(self):
        #print("\nCheckpoint")

        pulled_arm = 0
        max_upper_bound = 0
        total_counts = 0
        bound_length = 0

        for arm in range(0, self.n_arms):
            # if the arm arm has been already pulled once

            if self.selections_windowed[arm] > 0:
                total_counts = self.t
                # if total_counts > 20:
                #    total_counts = 20
                # after I reach the window size, I won't do more than 20 selections, so I can fix this number.
                bound_length = math.sqrt(0.01*math.log(self.t) / float(self.selections_windowed[arm]))
                upper_bound = self.empirical_mean[arm] + bound_length
                #print('mean {}'.format(self.empirical_mean[arm]))
                #print('bound {}'.format(bound_length))

            else:
                upper_bound = 1e400  # this happens just when arms haven't been pulled before, so
                # we give them an very large (inf) upper bounds

            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                pulled_arm = arm

            #print('a {}',upper_bound, self.empirical_mean[arm], bound_length, arm)
        #print(max_upper_bound)

        #if self.t % 100 == 0:
         #   print("\nNEW PHASE")
        #print("Number of selections", self.numbers_of_selections)
        #print("Total counts:", total_counts)
        #print("Pulled arm: ", pulled_arm)
        #print("Arms pulled", self.pulled_arms[-self.window_size:])
        #print(len(self.pulled_arms))
        #print("\nCheckpoint - pulling arm ", pulled_arm)
        #print("Pulled arm", pulled_arm)

        #print("Selections per arm", self.selections_windowed)
        #print("Selections for the pulled arm:", self.selections_windowed[pulled_arm])
        #print(total_counts)

        return pulled_arm

    def update(self, pulled_arm, reward):

        #print("\nCHECKPOINT - round ", self.t)
        self.t += 1
        self.update_observations(pulled_arm, reward)

        self.numbers_of_selections[pulled_arm] = self.numbers_of_selections[pulled_arm] + 1

        #print("Selections", self.numbers_of_selections)


        ##pulled arms contiene la sequenza di arm che son stati tirati
        self.pulled_arms = self.pulled_arms.astype(int)
        temp = np.bincount(self.pulled_arms[-self.window_size:], minlength=self.n_arms)
        #print(self.pulled_arms[-self.window_size:])
        #print("Selection per arm", np.bincount(self.pulled_arms[-160:], minlength=4))

        self.selections_windowed = temp
        #print("Selection per arm", self.selections_windowed)
        #print("Arm ", pulled_arm, " has been pulled ", self.selections_windowed[pulled_arm],
         #     " times in the last 20 rounds")

        num_selections_pulled_arm = self.selections_windowed[pulled_arm]

        # overall is the sum of the last 'num_selections_pulled_arm' rewards
        # where 'num_selections_pulled_arm' is the number of times the arm has been pulled in the last 20 pulls
        overall = np.sum(self.rewards_per_arm[pulled_arm][-num_selections_pulled_arm:])
        size = len(self.rewards_per_arm[pulled_arm][-num_selections_pulled_arm:])
        #print("Rewards for the pulled arm", self.rewards_per_arm[pulled_arm][-self.window_size:])

        #print("Last 20 arms pulled", self.pulled_arms[-20:])
        #print(self.rewards_per_arm[pulled_arm][-self.window_size:])

        #print("Overall", overall)
        #print("Size", size)
        #print('\n')
        #update the empirical mean taking into account only the last N rewards where N = self.window_size
        self.empirical_mean[pulled_arm] = overall/size
