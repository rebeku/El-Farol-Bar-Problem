import numpy as np

def err(hist, a):
    [c_4, c_3, c_2, c_1] = hist
    return (c_3 - c_1 + (c_2 - c_3) * a)**2 + (c_4 - c_2 + (c_3 - c_4) * a)**2


def pred(hist, a):
    [c_4, c_3, c_2, c_1] = hist
    return a * c_1 + (1 - a) * c_2


def assert_equal(a, b, msg):
    assert (np.abs(a - b) < 1e-6).all(), msg
    

def assert_less(a, b, msg):
    assert (a - b < 1e-6).all(), msg

    
def assert_geq(a,b,msg):
    assert (a - b > -1e-6).all(), msg


class BarModel:
    def __init__(
        self,
        hist, 
        s, 
        x_min, 
        x_max
    ):
        try:
            self.s = s
            self.x_min = x_min
            self.x_max = x_max
            self.predict(hist, s, x_min, x_max)
        except AssertionError as E:
            self.error = E
            print(E)
        
    def predict(self, hist, s, x_min, x_max):
        # write the errors as a quadratic eqation
        [c_4, c_3, c_2, c_1] = hist

        self.a_1 = (c_2 - c_3)**2 + (c_3 - c_4)**2
        self.a_2 = 2 * ((c_2 - c_3) * (c_3 - c_1) + (c_3 - c_4) * (c_4 - c_2))
        self.a_3 = (c_3 - c_1)**2 + (c_4 - c_2)**2

        # check work
        x = np.arange(-1, 1.01, 0.01)
        y = err(hist, x)
        y_2 = (self.a_1 * x**2 + self.a_2 * x + self.a_3)

        assert_equal( (y_2 - y).max(), 0, "Check your coefficients!")
        
        # check for degenerate  case
        # when prediction is a constant
        # equal to c_1
        if c_1 == c_2:
            if c_1 < 60:
                self.p_go = 1
                self.case = 1
            else:
                self.p_go = 0
                self.case = 4
            return
        
        # threshold
        self.t = (60 - c_2) / (c_1 - c_2)

        # whether or not prediction is increasing
        # determines which regions of the map are going
        # this inequality follows from the predictor
        # for time t
        self.increasing = c_1 > c_2
        
        # threshold lower than entire window
        if self.t <= x_min:
            if self.increasing:
                self.p_go = 0
                self.case = 5
            else:
                self.p_go = 1
                self.case = 2
            return
        
        # threshold above entire window
        if self.t >= x_max:
            if self.increasing:
                self.p_go = 1
                self.case = 3
            else:
                self.p_go = 0
                self.case = 6
            return
        
        
        self.q = (self.t - x_min) / (x_max - x_min)
        
        # when err function is a constant
        # when a_1 is 0, a_2 should also be 0
        # so this is convenient
        if self.a_1 == 0:
            assert self.a_2 == 0
    
            if self.increasing:
                self.p_go = self.q
                self.case = 7
            else:
                self.p_go = 1 - self.q
                self.case = 8
            return
            
        # optimal A is min of error curve
        self.opt_a = -self.a_2 / (2*self.a_1)

        assert_geq (y.min(), err(hist, self.opt_a), "Opt_a isn't minimum!")

        
        # is the bottom part going
        inc_going = self.increasing and self.t > self.opt_a
        dec_going = not self.increasing and self.t <= self.opt_a
        self.bottom_part_going = inc_going or dec_going
        
        # find coordinates of bottom region
        # reflection of threshold about *opt_a*
        self.t_ref  = 2 * self.opt_a - self.t

        assert_equal(self.opt_a - self.t_ref, self.t - self.opt_a, "reflected thresholds not symmetrical")
        
        # to get the bottom part we need the left-hand
        # and right-hand sides of the threshold
        self.t_l = min(self.t, self.t_ref)
        self.t_r = max(self.t, self.t_ref)

        assert_equal(self.opt_a - self.t_l, self.t_r - self.opt_a, "right and left thresholds not symmetrical")
        
        # reflection of LHS of window across threshold
        # this is the righthand boundary of the symmetric region
        self.ref_r = 2 * self.opt_a - x_min

        assert_equal(self.opt_a - x_min, self.ref_r - self.opt_a, "ref_r not symmetrical")

        # reflection of RHS of window across threshold
        # this is the lefthand boundary of the symmetric region
        self.ref_l = 2 * self.opt_a - x_max

        assert_equal(x_max - self.opt_a, self.opt_a - self.ref_l, "ref_l not symmetrical")

        
        # For the remaining cases we can now
        # assume that threshold is inside of
        # the window

        # threshold to the left of all points other 
        # interesting things.
        # agents will always pick a > t
        # when possible
        if self.t < self.ref_l:
            # the probability that any is in the optimal region
            if self.increasing:
                self.p_go = self.q**s
                self.case = 10
            else:
                self.p_go = 1 - self.q**s
                self.case = 9
            return
        
        # threshold within window
        # but to the right of all points other 
        # interesting things.
        # agents will always pick a <= t
        # when possible        
        if self.t > self.ref_r:
            if self.increasing:
                self.p_go = 1 - (1 - self.q)**s
                self.case = 11
            else:
                self.p_go = (1 - self.q)**s
                self.case = 12
            return
                
        # the remaining cases have a bottom region where the threshold
        # is closer to opt_a than either reflected boundary.  The bottom
        # region is the reflection of the threshold about opt_a.  The symmetric
        # region is the area between the reflected threshold and reflected
        # boundaries, where there is no preference between strategies
        # that attend the bar and those that do not.
        self.p_anything_bottom = 1 - (1 - 2 * np.abs(self.t - self.opt_a) / (x_max - x_min))**s
        
        # asymmetric region is on left
        if x_min <= self.ref_l:
            self.p_all_left_asym = ((self.ref_l - x_min) / (x_max - x_min))**s
            self.p_sym = 1 - self.p_anything_bottom - self.p_all_left_asym
            
            assert_geq(self.p_sym, 0, f"Negative probability of {self.p_sym}!  Oh no!")
            
            # bottom goes, left side doesn't
            if self.t < self.opt_a and not self.increasing:
                self.case = 13
                self.p_go = self.p_anything_bottom + 0.5 * self.p_sym
                
            # bottom and left side go
            elif self.t > self.opt_a and self.increasing:
                self.case = 14
                self.p_go = 1 - 0.5 * self.p_sym
                
            # only symmetric region can go
            elif self.t > self.opt_a:
                self.case = 15
                self.p_go = 0.5 * self.p_sym
                
            # left side goes, bottom does not
            else:
                self.case = 16
                self.p_go = 0.5 * self.p_sym + self.p_all_left_asym
                
            return
                
        if x_max > self.ref_r:
            self.p_all_right_asym = ((x_max - self.ref_r) / (x_max - x_min))**s
            self.p_sym = 1 - self.p_anything_bottom - self.p_all_right_asym
            
            assert_geq(self.p_sym, 0, f"Negative probability of {self.p_sym}!  Fiddlesticks!")
            
            # bottom goes, right side doesn't
            if self.t > self.opt_a and self.increasing:
                self.p_go = self.p_anything_bottom + 0.5 *self.p_sym
                self.case = 17
                
            # bottom and right side go
            elif self.t < self.opt_a and not self.increasing:
                self.p_go = 1 - 0.5 * self.p_sym
                self.case = 18
                
            # right side goes, bottom doesn't
            elif self.t > self.opt_a and not self.increasing:
                self.p_go = self.p_all_right_asym + 0.5 * self.p_sym
                self.case = 19
                
            # only symmetric region can go
            elif self.t < self.opt_a and self.increasing:
                self.p_go = 0.5 * self.p_sym
                self.case = 20
                
            else:
                assert False, "WTF???"
            return
        else:
            assert False, "Missing case!"
            
    def iterate(self, start, agents, n):
        hist = np.zeros(shape=n + 4, dtype=int)
        cases = np.zeros(shape=n)
        
        hist[:4] = start
        
        for i in range(4, n+4):
            self.predict(hist[i-4:i], self.s, self.x_min, self.x_max)
            hist[i] = int(self.p_go * agents)
            cases[i - 4] = self.case
            
        return hist, cases