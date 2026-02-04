# Linear Regression Course - Helper Functions

**Visualization and utility functions for INFO 7390: Understanding Data (Advanced Data Science)**  
**Northeastern University | Master's in Information Systems**

---

## 📚 Overview

This repository contains helper functions for a 3-part linear regression series taught using Google Colab notebooks:

1. **Notebook 1:** Simple Linear Regression - Intuition & Mechanics
2. **Notebook 2:** Inference, Uncertainty & Model Fit  
3. **Notebook 3:** Multiple Regression, Diagnostics & Extensions

The helper functions provide clean visualizations and demonstrations without cluttering the teaching notebooks with implementation details.

---

## 🚀 Quick Start
### Using These Helpers in Google Colab

Add this cell at the **top of your notebook**:
```python
#@title **⚡ Setup - Run This First!** { display-mode: "form" }

import urllib.request

# Download appropriate helper file
# For Notebook 1:
url = 'https://raw.githubusercontent.com/YOUR-USERNAME/data-science-course-helpers/main/nb1_helpers.py'

# For Notebook 2:
# url = 'https://raw.githubusercontent.com/YOUR-USERNAME/data-science-course-helpers/main/nb2_helpers.py'

# For Notebook 3:
# url = 'https://raw.githubusercontent.com/YOUR-USERNAME/data-science-course-helpers/main/nb3_helpers.py'

urllib.request.urlretrieve(url, 'helpers.py')
from helpers import *

print("✅ Helper functions loaded!")
```

**That's it!** Now you can use the functions throughout your notebook.

---

## 🔧 Requirements

### Python Version
- Python 3.7+

### Required Packages
```
numpy>=1.19.0
pandas>=1.1.0
matplotlib>=3.3.0
seaborn>=0.11.0
scipy>=1.5.0
statsmodels>=0.12.0
scikit-learn>=0.24.0
```

### Install All Dependencies
```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn
```

**In Colab:** All packages are pre-installed! ✅

---

## Contribution

Anusha Prakash
Teaching Assistant
