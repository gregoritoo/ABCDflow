from search import preparekernel,search_and_add,prune,replacekernel
import unittest 

class preparekernelTest(unittest.TestCase):
    def test_preparekernel(self):
        kernel_list = ["+PER","*LIN"] 
        kernels = preparekernel(kernel_list)
        dic = {"parameters_per":["periodic_l","periodic_p","periodic_sigma"],"parameters_lin":["lin_c","lin_sigmav"]}
        assert kernels == dic

    def test_preparekernel2(self) :
        kernel_list = ["+PER","+LIN"] 
        kernels = preparekernel(kernel_list)
        dic = {"parameters_per":["periodic_l","periodic_p","periodic_sigma"],"parameters_lin":["lin_c","lin_sigmav"]}
        assert kernels == dic

    def test_preparekernel3(self) :
        kernel_list = ["+PER","+LIN","*SE"] 
        kernels = preparekernel(kernel_list)
        dic = {"parameters_per":["periodic_l","periodic_p","periodic_sigma"],"parameters_lin":["lin_c","lin_sigmav"],"parameters_se":["squaredexp_l","squaredexp_sigma"]}
        assert kernels == dic

class search_and_addTest(unittest.TestCase):
    def test_search_and_addTest(self):
        kernel_list = tuple(["+PER","+LIN"])
        output = search_and_add(kernel_list)
        res = [("+PER","+LIN","+LIN"),("+PER","+LIN","+PER"),("+PER","+LIN","+SE"),("+PER","+LIN","*LIN"),("+PER","+LIN","*SE"),("+PER","+LIN","*PER")]
        self.assertListEqual(sorted(output),sorted(res))

    def test_search_and_addTest2(self):
        kernel_list = tuple(["+LIN"])
        output = search_and_add(kernel_list)
        res = [("+LIN","+LIN"),("+LIN","+PER"),("+LIN","+SE"),("+LIN","*LIN"),("+LIN","*SE"),("+LIN","*PER")]
        self.assertListEqual(sorted(output),sorted(res))


class pruneTest(unittest.TestCase):
    def test_prune(self):
        tempbest = [["+LIN"]]
        rest = [("+LIN","+LIN"),("+LIN","+PER"),("+LIN","+SE"),("+LIN","*LIN"),("+LIN","*SE"),("+LIN","*PER"),
                ("+PER","+LIN"),("+PER","+PER"),("+PER","+SE"),("+PER","*LIN"),("+PER","*SE"),("+PER","*PER"),
                ("+SE","+LIN"),("+SE","+PER"),("+SE","+SE"),("+SE","*LIN"),("+SE","*SE"),("+SE","*PER")]
        new_rest = prune(tempbest,rest)
        res = [["+LIN","+LIN"],["+LIN","+PER"],["+LIN","+SE"],["+LIN","*LIN"],["+LIN","*SE"],["+LIN","*PER"]]
        self.assertListEqual(sorted(new_rest),sorted(res))

    def test_prune2(self):
        tempbest = [["+PER"]]
        rest = [("+LIN","+LIN"),("+LIN","+PER"),("+LIN","+SE"),("+LIN","*LIN"),("+LIN","*SE"),("+LIN","*PER"),
                ("+PER","+LIN"),("+PER","+PER"),("+PER","+SE"),("+PER","*LIN"),("+PER","*SE"),("+PER","*PER"),
                ("+SE","+LIN"),("+SE","+PER"),("+SE","+SE"),("+SE","*LIN"),("+SE","*SE"),("+SE","*PER")]
        new_rest = prune(tempbest,rest)
        res = [["+PER","+LIN"],["+PER","+PER"],["+PER","+SE"],["+PER","*LIN"],["+PER","*SE"],["+PER","*PER"]]
        self.assertListEqual(sorted(new_rest),sorted(res))


class replacekernelTest(unittest.TestCase):
    def test_replaceKernel(self):
        kernel_list = ["+PER","+LIN"] 
        output = [["+SE","+LIN"], ["+LIN","+LIN"] , ["+PER","+SE"] , ["+PER","+PER"] , ["+PER","+LIN"]  ]
        res = replacekernel(kernel_list)
        self.assertListEqual(sorted(output),sorted(res))
