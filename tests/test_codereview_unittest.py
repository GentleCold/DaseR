import unittest
import os

class TestCodeReview(unittest.TestCase):
    def test_dir(self):
        self.assertTrue(os.path.exists('codereview'), 'codereview directory does not exist')
        self.assertTrue(os.path.isdir('codereview'), 'codereview is not a directory')

if __name__ == '__main__':
    unittest.main()
