import unittest


class SimpleTest(unittest.TestCase):

    # Returns True or False.
    def test_all_indexes(self,set,list,indexes):
        if len(set) != len(list):
            print("SET AND LIST NOG EQUAL ")
            print("size set ", len(set))
            print("size list ", len(list))

       # self.assertEqual(len(set),len(indexes))

       # for i in indexes:
        #    self.assertTrue(indexes[i] in set)
