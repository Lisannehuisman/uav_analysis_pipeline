# Supplementary Associated Code

This folder keeps additional code and configuration files from the broader working project that may be useful for understanding or extending the thesis experiments. The main cleaned thesis workflow still lives in `src/`, `data_collection/`, and `results/`.

The files here preserve their original relative paths under `original_workspace/`. They include older or parallel scripts for detector comparisons, viewpoint analysis, multiview fusion, probability fusion, plotting, reporting, and the experimental multiview transformer work.

I have not copied the large generated outputs, raw public dataset dumps, model checkpoints, caches, virtual environments, or label-only text dumps into this folder. Some scripts may therefore need paths adjusted or omitted data restored before they can run. The point of this folder is to keep as much associated code as possible without making the GitHub repository enormous or mixing every exploratory script into the main thesis pipeline.

`associated_code_manifest.csv` lists each copied file and where it came from in the original workspace.
