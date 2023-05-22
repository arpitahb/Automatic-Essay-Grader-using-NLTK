import unittest
import app
import json

class MyTestCase(unittest.TestCase):
    def test_something(self):
        print("testing response.")
        path = "D:\\PROJECT\\venv\\event.json"
        with open(path, "r") as file:
             event = json.load(file)
        result = app.lambda_handler(event, None)
        print(result)
        self.assertEqual(result['statusCode'], 200)
        # self.assertEqual(result['headers']['Content-Type'], 'application/json')
        # self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
