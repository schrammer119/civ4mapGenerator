import ast
import os
import unittest


class Python24SyntaxTests(unittest.TestCase):
    def test_planetsim_avoids_conditional_expressions(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "PlanetSim.py")
        with open(source_path, "r") as source_file:
            syntax_tree = ast.parse(source_file.read(), filename=source_path)

        conditional_lines = [
            node.lineno for node in ast.walk(syntax_tree) if isinstance(node, ast.IfExp)
        ]
        self.assertEqual(
            [],
            conditional_lines,
            "Python 2.4 cannot parse conditional expressions.",
        )


if __name__ == "__main__":
    unittest.main()
