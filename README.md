# Quantum Tunnelling through a Rectangular Barrier — Streamlit

Faithful browser conversion of the supplied standalone Matplotlib desktop
laboratory.

## Preserved

- Same HBAR and MASS constants
- Same E, V0 and a ranges and step sizes
- Same wave-number calculation
- Same 4x4 complex continuity system
- Same piecewise wavefunction
- Same analytical transmission expression
- Same |t|, T=|t|², R=|r|² and R+T readout
- Same measurement-window concept
- Same instructor-key environment variable

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the Streamlit URL shown in the terminal, normally:
http://localhost:8501

## Streamlit Community Cloud

1. Create a GitHub repository.
2. Put `app.py` and `requirements.txt` in the repository root.
3. Open Streamlit Community Cloud.
4. Select the repository and `app.py`.
5. Deploy.

## Instructor key

For local use, set:

Windows CMD:
```bat
set TUNNELING_INSTRUCTOR=1
streamlit run app.py
```

PowerShell:
```powershell
$env:TUNNELING_INSTRUCTOR="1"
streamlit run app.py
```

The browser version uses Plotly because Matplotlib's desktop Slider GUI
cannot run as a native interactive window inside Streamlit.
