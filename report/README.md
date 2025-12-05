# ANPR Project Report

## 📄 Report Files

- `ANPR_Report.tex` - IEEE format LaTeX source (compile to get PDF)
- `ANPR_Report_Summary.md` - Quick summary for reference

## 🔧 How to Compile LaTeX to PDF

### Option 1: Overleaf (Recommended - Easy)
1. Go to [overleaf.com](https://www.overleaf.com)
2. Create new project → Upload Project
3. Upload `ANPR_Report.tex`
4. Click "Recompile" → Download PDF

### Option 2: Local Compilation (Mac)
```bash
# Install MacTeX if not installed
brew install --cask mactex

# Compile
cd report
pdflatex ANPR_Report.tex
pdflatex ANPR_Report.tex  # Run twice for references
```

### Option 3: VS Code
1. Install "LaTeX Workshop" extension
2. Open `ANPR_Report.tex`
3. Press `Cmd+Alt+B` to build

## 📊 Key Results Summary

| Metric | YOLOv5 | YOLOv9-GELAN |
|--------|--------|--------------|
| mAP@0.5 | **86.73%** | 74.92% |
| mAP@0.5:0.95 | **46.95%** | 26.95% |
| Precision | 85.3% | **91.8%** |
| Recall | 82.0% | **95.9%** |
| Parameters | **7.0M** | 25.5M |
| Speed (CPU) | **45ms** | 120ms |

## 📑 Report Structure

1. **Abstract** - Overview of ANPR system and results
2. **Introduction** - Background and contributions
3. **Related Work** - Traditional and deep learning methods
4. **Problem Statement** - Challenges in ANPR
5. **Methodology** - YOLOv5/YOLOv9 architectures + OCR
6. **Experimental Results** - Comparative metrics and analysis
7. **Discussion** - Advantages of each model
8. **Conclusion** - Summary and future work
9. **References** - 9 academic citations

