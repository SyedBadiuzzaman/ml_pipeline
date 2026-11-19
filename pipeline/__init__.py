"""HiveSpace desk occupancy pipeline package.

Keep this package init lightweight so importing `pipeline.process_split` inside
the production Docker container does not pull in training-only packages like
`scikit-learn` or `matplotlib`.

To run the full pipeline from the project root:
    python pipeline/init.py
"""
