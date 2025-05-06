import re
import xml.etree.ElementTree as ET
import pandas as pd
from collections import Counter

def create_snippet_table(xml_path: str, output_html_path: str) -> None:
    """
    Parse an XML and generate an HTML table showing each snippet for
    each patient, with color coded diagnosis badges.
    """
    # Parse XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract patient IDs and their diagnosis code lists
    rows = []
    for snippet in root.findall('snippet'):
        # Expect snippet IDs like "123(45)" -> patient=123, doc=45
        sid = snippet.get('id')
        m = re.match(r"(?P<patient>\d+)\((?P<doc>\d+)\)", sid)
        if not m:
            continue
        patient_id = int(m.group('patient'))
        # Collect ICd-10 Codes
        codes = sorted(
            diag.get('icd10Code')
            for diag in snippet.find('Tumordiagnosen').findall('Diagnose')
            if diag.get('icd10Code')
        )
        rows.append({'patient_id': patient_id, 'codes': codes})

    # Build DataFrame from the extracted data
    df = pd.DataFrame(rows)

    # Identify all unique diagnosis codes across all patients
    all_codes = sorted({code for codes in df['codes'] for code in codes})
    '''
    print(f"Found {len(all_codes)} unique diagnosis codes: {', '.join(all_codes)}")

    # Compute frequency of each code across all snippets
    code_list = [code for codes in df['codes'] for code in codes]
    freq = Counter(code_list)
    print("Code frequencies:")
    for code in all_codes:
        print(f"  {code}: {freq[code]}")
    '''

    # Prepare color palette and map each code to a color
    palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    color_map = {code: palette[i % len(palette)] for i, code in enumerate(all_codes)}

    # Define two light grey tones for alternation
    group_colors = ['#f9f9f9', '#ebebeb']

    # Count how many snippets each patient has, and add snippet_index column
    counts = df.groupby('patient_id').size().to_dict()
    df['snippet_index'] = range(0, len(df))

    # Build HTML table rows
    html_rows = []
    group_idx = 0
    for patient_id, group in df.groupby('patient_id'):
        bg_color = group_colors[group_idx % len(group_colors)]
        total = counts[patient_id]
        first = True
        for _, row in group.iterrows():
            # Create colored badge spans for each diagnosis code
            diag_html = ''.join(
                f"<span style='background-color:{color_map[code]};"
                "padding:2px 6px;margin:2px;border-radius:4px;"
                "color:#fff;display:inline-block;font-size:0.9em;'>"
                f"{code}</span>"
                for code in row['codes']
            ) or '&nbsp;'

            if first:
                # For first snippet of each patient, include rowspan cell
                html_rows.append(
                    f"<tr style='background-color:{bg_color};'>"
                    f"<td rowspan='{total}'>{total}</td>"
                    f"<td>{row['snippet_index']}</td>"
                    f"<td style='text-align:left;'>{diag_html}</td>"
                    f"</tr>"
                )
                first = False
            else:
                # Subsequent rows don’t repeat the rowspan cell
                html_rows.append(
                    f"<tr style='background-color:{bg_color};'>"
                    f"<td>{row['snippet_index']}</td>"
                    f"<td style='text-align:left;'>{diag_html}</td>"
                    f"</tr>"
                )
        group_idx += 1

    # Assemble full HTML document
    html_output = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Snippet-Tabelle</title>
  <style>
    table {{
      border-collapse: collapse;
      margin: 20px auto;
      width: auto;
      table-layout: fixed;
    }}
    colgroup col:nth-child(1) {{ width: 33%; }}
    colgroup col:nth-child(2) {{ width: 33%; }}
    colgroup col:nth-child(3) {{ width: 34%; }}
    th, td {{
      border: 1px solid #ccc;
      padding: 4px;
      text-align: center;
      vertical-align: middle;
      word-wrap: break-word;
    }}
    th {{
      background: #f2f2f2;
    }}
  </style>
</head>
<body>
  <table>
    <colgroup>
      <col>
      <col>
      <col>
    </colgroup>
    <thead>
      <tr>
        <th>Patient #Snippets</th>
        <th>Snippet-ID</th>
        <th>Diagnoses</th>
      </tr>
    </thead>
    <tbody>
      {''.join(html_rows)}
    </tbody>
  </table>
</body>
</html>
"""
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_output)


create_snippet_table('evalset.xml', 'snippet_table.html')
