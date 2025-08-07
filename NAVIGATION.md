# Navigation Structure

The Week 1 fundamentals content is now fully integrated into the Hugo site with the following navigation structure:

## Main Navigation Path
```
Homepage (/)
└── Syllabus (/docs/)
    └── Module 1: Fundamentals (/docs/fundamentals/)
        └── Week 1: Python Fundamentals (/docs/fundamentals/week1/)
            ├── Python Basics (/docs/fundamentals/week1/python-basics/)
            ├── NumPy Arrays (/docs/fundamentals/week1/numpy-arrays/)
            ├── Pandas DataFrames (/docs/fundamentals/week1/pandas-dataframes/)
            └── Matplotlib Visualization (/docs/fundamentals/week1/matplotlib-visualization/)
```

## Direct Links to Content

### Week 1 Overview
- **URL**: `/docs/fundamentals/week1/`
- **Description**: Complete overview of Week 1 with learning objectives, setup instructions, and projects

### Individual Topics
1. **Python Basics for DevOps/SRE**
   - URL: `/docs/fundamentals/week1/python-basics/`
   - Content: Core Python with infrastructure automation examples

2. **NumPy for Infrastructure Metrics**
   - URL: `/docs/fundamentals/week1/numpy-arrays/`
   - Content: Efficient numerical operations for metrics analysis

3. **Pandas for Log Analysis**
   - URL: `/docs/fundamentals/week1/pandas-dataframes/`
   - Content: Log analysis and infrastructure data management

4. **Matplotlib for Monitoring Dashboards**
   - URL: `/docs/fundamentals/week1/matplotlib-visualization/`
   - Content: Creating professional monitoring visualizations

## Running the Site

```bash
# Start the Hugo development server
hugo server

# View the site at
http://localhost:1313

# Navigate to Week 1 content
http://localhost:1313/docs/fundamentals/week1/
```

## File Locations

- **Hugo Content**: `/content/docs/fundamentals/week1/`
- **Original Content**: `/fundamentals/week1/` (preserved as backup)

All content includes:
- Real-world DevOps/SRE examples
- Practical code samples
- Hands-on exercises
- Comprehensive references with links

The navigation is fully functional with the Hextra theme's built-in search and responsive design.