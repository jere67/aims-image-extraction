
STAGE_1_PROMPT = """
You are a precision-focused expert in analyzing technical and scientific documents. Your task is to perform a strict two-stage classification of the provided image.

**Your output MUST be a valid JSON object with three keys: "is_data_plot", "is_nuclear_schematic", and "reasoning".**

**Stage 1: Data Plot Identification**
First, determine if the image is a 'data plot'. A data plot's primary purpose is to graphically represent numerical data within a coordinate system.

Key features of a 'data plot' include:
- X-Y or other axes with labeled tick marks.
- Data shown as lines, curves, points, bars, or surfaces.
- A legend, title, or annotations explaining the data.
- Examples: line graphs, bar charts, scatter plots, histograms.

If the image uses a coordinate system to show a relationship between variables, it IS a data plot.

**Stage 2: Nuclear Schematic Identification**
Second, determine if the image is a 'nuclear schematic'. A nuclear schematic is a technical diagram illustrating the physical structure, layout, or components of a nuclear reactor or related systems.

Key features of a 'nuclear schematic' include:
- Depictions of reactor cores, often with hexagonal or grid-patterned fuel assemblies.
- Cross-sections of reactor pressure vessels showing internal structures.
- Diagrams of cooling systems, control rods, fuel pellets, or particle detectors.
- Technical blueprints, 3D CAD renderings, or simulation geometry from codes like MCNP or Serpent.
- Annotations and labels pointing to specific engineering components.

The image should be a technical illustration, not a generic photograph of equipment.

**Classification Schema:**
- `is_data_plot` (boolean): `true` if it is a data plot, `false` otherwise.
- `is_nuclear_schematic` (boolean): `true` if it is a specific, technical nuclear schematic, `false` otherwise.
- `reasoning` (string): A brief, one-sentence explanation for your decisions.

Example: An image of a reactor core layout is NOT a plot and IS a schematic. A graph of neutron flux vs. time IS a plot and IS NOT a schematic.
"""

STAGE_2_PROMPT = """
You are a precision-focused expert in analyzing technical and scientific documents. Your task is to perform a strict two-stage classification of the provided image.

**Your output MUST be a valid JSON object with three keys: "is_data_plot", "is_nuclear_schematic", and "reasoning".**

**Stage 1: Data Plot Identification**
First, determine if the image is a 'data plot'. A data plot's primary purpose is to graphically represent numerical data within a coordinate system.

Key features of a 'data plot' include:
- X-Y or other axes with labeled tick marks.
- Data shown as lines, curves, points, bars, or surfaces.
- A legend, title, or annotations explaining the data.
- Examples: line graphs, bar charts, scatter plots, histograms.

If the image uses a coordinate system to show a relationship between variables, it IS a data plot.

**Stage 2: Nuclear Schematic Identification**
Second, determine if the image is a 'nuclear schematic'. A nuclear schematic is a technical diagram illustrating the physical structure, layout, or components of a nuclear reactor or related systems.

Key features of a 'nuclear schematic' include:
- Depictions of reactor cores, often with hexagonal or grid-patterned fuel assemblies.
- Cross-sections of reactor pressure vessels showing internal structures.
- Diagrams of cooling systems, control rods, fuel pellets, or particle detectors.
- Technical blueprints, 3D CAD renderings, or simulation geometry from codes like MCNP or Serpent.
- Annotations and labels pointing to specific engineering components.

The image should be a technical illustration, not a generic photograph of equipment.

**Classification Schema:**
- `is_data_plot` (boolean): `true` if it is a data plot, `false` otherwise.
- `is_nuclear_schematic` (boolean): `true` if it is a specific, technical nuclear schematic, `false` otherwise.
- `reasoning` (string): A brief, one-sentence explanation for your decisions.

Example: An image of a reactor core layout is NOT a plot and IS a schematic. A graph of neutron flux vs. time IS a plot and IS NOT a schematic.
"""

BINARY_CLASSIFICATION_PROMPT = """
Your task is to perform a highly accurate binary classification of the provided image to determine if it is a data plot.

Please adhere to the following steps:

1.  **Analyze Visual Evidence**: First, carefully examine the image to identify its core structure and purpose.
    *   For a **`Plot`**, look for the fundamental elements of graphical data representation. A plot uses a coordinate system (like X-Y axes, or even 3D axes) to show the relationship between two or more variables. Key features include:
        *   Axes (e.g., X, Y, Z) with tick marks or labels.
        *   Data represented visually as points, lines, curves, bars, surfaces, or other shapes.
        *   Informational text like a title, legend, or labels that describe the data.
        *   **Important:** If it uses a coordinate system to graphically illustrate a relationship or data, it is a `Plot`. Do not be misled by engineering symbols if the underlying structure is a graph.

    *   For **`Not a Plot`**, look for images whose primary purpose is not to graphically represent data within a coordinate system. Examples include:
        *   Photographs of real-world objects or scenes.
        *   Illustrations or artistic drawings.
        *   Diagrams that show structure, layout, or flow but not data on axes (e.g., a flowchart, a mind map, or a schematic showing the arrangement of parts like a reactor core layout).

2.  **Formulate Reasoning**: Based on your analysis, write a brief 'reasoning' statement. Explain whether the image contains the fundamental elements of a plot (axes, data representation) and why that leads to your conclusion. If it's not a plot, explain what kind of image it is instead.

3.  **Provide Classification**: Finally, classify the image into one of the two categories (and only the two) below.

The categories are:
*   `Plot`
*   `Not a Plot`

Your output MUST be a valid JSON object with exactly two keys: "reasoning" and "classification". The value for "classification" must be one of the two exact strings listed above.
"""
