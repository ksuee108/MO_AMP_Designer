import streamlit as st
import pandas as pd
from design import  algorithms_setup, plot_pareto_fronts_many, plot_pareto_fronts_multi, amino_acid_percentage
import os
from BioAnalysis import Bio_analysis
from Bio import SeqIO
from io import StringIO
from model import CNNModel, AutoEncoderRegressor, DenoisingAutoEncoderRegressor, VariationalAutoEncoderRegressor, ContrastiveEncoder
import torch
import peptides
import numpy as np
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = joblib.load("model/scaler.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")
user_home = os.path.expanduser("~")

st.set_page_config(page_title="MO-AMP designer", layout="wide")
st.title("🧬 MO-AMP designer")
st.text("This app allows users to explore and design antimicrobial peptides by multi-objective optimization.")
main_tab1, main_tab2, main_tab3, main_tab4, main_tab5 = st.tabs(["Home", "About this App", "Method Overview", "How to Design", "Related Databases and Prediction Websites"])

footer = """
<style>
/* 為固定在底部的 footer 預留空間,避免擋住內容與按鈕 */
[data-testid="stMainBlockContainer"],
.block-container {
    padding-bottom: 90px !important;
}
[data-testid="stSidebarUserContent"] {
    padding-bottom: 90px !important;
}

.footer-text {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background: rgb(240,240,240);
    color: black;
    text-align: center;
    padding: 10px 0;
    font-size: 14px;
    border-top: 1px solid #ccc;
    z-index: 100;   /* 降低層級,避免蓋住下拉選單與對話框 */
}
</style>

<div class="footer-text">
    🚀 MO-AMP Design App © 2026<br>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

def is_valid_peptide(seq: str) -> bool:
    return len(seq) > 0 and set(seq.upper()) <= VALID_AA

with main_tab1:
    # ----------------------------
    # Sidebar inputs
    # ----------------------------
    all_features = []
    with st.sidebar:
        st.header("Select Bacteria")
        Bacteria = st.multiselect(
            "Choose bacteria",
            options=["E. faecium", "S. aureus", "K. pneumoniae", "A. baumannii", "P. aeruginosa", "E. coli", "B. subtilis", "P. vulgaris"]
        )

        st.header("Upload Peptide Sequence")
        uploaded_file = st.file_uploader("Upload FASTA", type=["fasta", "fa"])

        seed = st.number_input("Random seed", min_value=0, max_value=10000, value=42, step=1)
        seq = None
        bio_analysis = None

        if uploaded_file is not None:
            file_type = uploaded_file.name.lower()

            uploaded_file.seek(0)
            fasta_str = uploaded_file.read().decode("utf-8")
            fasta_io = StringIO(fasta_str)

            records = list(SeqIO.parse(fasta_io, "fasta"))

            if len(records) == 0:
                st.error("FASTA file is empty or invalid.")
            else:
                for rec in records:
                    seq = str(rec.seq).strip().upper()
                    if not is_valid_peptide(seq):
                        st.warning(f"Skipped {rec.id}: contains non-standard amino acids.")
                        continue
                    
                    try:
                        bio_analysis = Bio_analysis(seq)

                        Gravy = bio_analysis.get_gravy()
                        instability_index = bio_analysis.get_instability_index()
                        Aliphatic_Index = bio_analysis.get_aliphatic_index()
                        Boman_index = bio_analysis.get_boman_index()
                        isoelectric_point = bio_analysis.get_isoelectric_point()
                        net_charge = bio_analysis.get_net_charge()
                        molecular_weight = bio_analysis.get_molecular_weight()
                        charge_at_pH = bio_analysis.get_charge_at_pH()
                        aromaticity = bio_analysis.get_aromaticity()
                        sec_H, sec_T, sec_S = bio_analysis.get_secondary_structure_fraction()
                        amphipathicity = bio_analysis.get_amphipathicity()
                        correlation = bio_analysis.get_auto_correlation()
                        covariance = bio_analysis.get_auto_covariance()
                        hydrophobic_moenet = bio_analysis.get_hydrophobic_moenet()
                        mass = bio_analysis.get_mass()
                        mz = bio_analysis.get_mz()
                        SequenceLength = bio_analysis.get_sequenceLength()

                        features = {
                            "Sequence": seq,
                            'Length': SequenceLength,
                            'Gravy': Gravy,
                            'Instability Index': instability_index,
                            'Aliphatic Index': Aliphatic_Index,
                            'Isoelectric point': isoelectric_point,
                            'Net charge': net_charge, 
                            'Molecular Weight': molecular_weight,
                            'Charge at pH': charge_at_pH,
                            'Aromaticity': aromaticity,
                            'Secondary structure fraction Helix': sec_H,
                            'Secondary structure fraction Turn': sec_T,
                            'Secondary structure fraction Sheet': sec_S,
                            'Boman Index': Boman_index,
                            'Amphipathicity': amphipathicity,
                            'Correlation': correlation,
                            'Covariance': covariance,
                            'Mass': mass,
                            'Mz': mz,
                        }
                        all_features.append(features)

                    except Exception as e:
                        st.error(f"Error parsing Sequence {rec.id}: {e}")
        
        all_features = pd.DataFrame(all_features)

        # ----------------------------
        # Optimization settings
        # ----------------------------
        st.markdown("---")
        st.header("Select Algorithms")
        algorithms = st.multiselect(
            "Choose optimization algorithms",
            options=[
                'NSGA-II', 'NSGA-III', 'R-NSGA-II', 'R-NSGA-III',
                'U-NSGA-III', 'AGE-MOEA', 'AGE-MOEA-II'
            ]
        )

        pop_size = st.number_input("Population size", min_value=10, max_value=400, value=80, step=10)
        length = st.number_input("Peptide Sequence length", min_value=10, max_value=50, value=10, step=1)
        generations = st.number_input("Number of generations", min_value=10, max_value=200, value=100, step=1)
        
    # ----------------------------
    # Objective selection
    # ----------------------------
    st.header("Objectives to optimize")
    opt = st.multiselect(
        "Select properties to optimize",
        options=[
            'Gravy', 'Instability Index', 'Aliphatic Index', 'Isoelectric point',
            'Net charge', 'Molecular Weight', 'Charge at pH', 'Aromaticity',
            'Secondary structure fraction Helix', 'Secondary structure fraction Turn',
            'Secondary structure fraction Sheet', 'Boman Index'
        ]
    )
    can_proceed = True
    if len(Bacteria) == 0 and uploaded_file is None:
        st.warning("Please select at least one Bacteria or upload a dataset.")
        can_proceed = False

    if len(opt) < 2:
        st.warning("Please select at least two objectives to optimize.")
        can_proceed = False

    if len(algorithms) < 1:
        st.warning("Please select at least one algorithm to optimize.")
        can_proceed = False
    else:
        optimization_directions = {}
        for i in opt:
            if i != "Gravy" :
                optimization_directions[i] = st.selectbox(
                    f"Select minima or maxima for {i}",
                    options=['Minimize', 'Maximize'],
                    key=i
                )
            else:
                optimization_directions[i] = st.selectbox(
                    f"Select optimiz hydrophobicity or hydrophilicity for {i}",
                    options=['hydrophobicity', 'hydrophilicity'],
                    key=i
                )
        st.success("Optimization Directions Set:")
        st.json(optimization_directions)

        # ----------------------------
        # Reference point (R-NSGA only)
        # ----------------------------
        ref_targets = None
        if any(a in algorithms for a in ("R-NSGA-II", "R-NSGA-III")):
            st.markdown("---")
            st.header("Reference point (R-NSGA)")
            st.caption(
                "Enter the ideal value you want for each objective, in the original "
                "units (e.g. Net charge = 5, not -5). R-NSGA-II/III will concentrate "
                "the search around this point."
            )
            ref_targets = {}
            for name in opt:
                ref_targets[name] = st.number_input(
                    f"Target for {name} ({optimization_directions[name]})",
                    value=0.0,
                    step=0.1,
                    format="%.2f",
                    key=f"ref_{name}"
                )
        # ----------------------------
        # Constraints
        # ----------------------------
        st.markdown("---")
        st.header("Constraints")

        options=[
                'Gravy', 'Instability Index', 'Aliphatic Index', 'Isoelectric point',
                'Net charge', 'Molecular Weight', 'Charge at pH', 'Aromaticity',
                'Secondary structure fraction Helix', 'Secondary structure fraction Turn',
                'Secondary structure fraction Sheet', 'Boman Index'
            ]

        if "constraints" not in st.session_state:
            st.session_state.constraints = []

        constraint_feature = st.selectbox("Select a physicochemical property:", options)
        constraint_type = st.radio(
            "Constraint type:",
            [f"{constraint_feature} ≤ (Maximum limit)",
             f"{constraint_feature} ≥ (Minimum limit)"]
        )
        constraint_value = st.number_input(
            f"Enter limit value for {constraint_feature}:",
            value=0.0,
            step=0.1,
            format="%.2f"
        )

        if st.button("➕ Add Constraint"):
            new_constraint = {
                "Feature": constraint_feature,
                "Type": "max" if "≤" in constraint_type else "min",
                "Value": constraint_value,
            }

            existing = [c for c in st.session_state.constraints if c["Feature"] == constraint_feature]
            if existing:
                st.warning(f"⚠️ {constraint_feature} constraint already exists — updated value.")
                st.session_state.constraints = [
                    new_constraint if c["Feature"] == constraint_feature else c
                    for c in st.session_state.constraints
                ]
            else:
                st.session_state.constraints.append(new_constraint)
                st.success(f"✅ Added: {constraint_feature} ({new_constraint['Type']} = {constraint_value})")

        if st.button("🗑️ Clear All Constraints"):
            st.session_state.constraints = []
            st.info("All constraints have been cleared.")

        if st.session_state.constraints:
            st.subheader("Current Constraints")
            df_constraints = pd.DataFrame(st.session_state.constraints)
            st.dataframe(df_constraints, use_container_width=True)

        constraint_dict_list = st.session_state.constraints
        st.write(constraint_dict_list)

        # ----------------------------
        # Load data based on selected bacteria
        # ----------------------------
        st.markdown("---")

        upload_name = uploaded_file.name if uploaded_file is not None else None
        cache_key = (tuple(sorted(Bacteria)), upload_name)

        if st.session_state.get("loaded_key") != cache_key:
            dfs = []

            # 1) user-uploaded FASTA file
            if uploaded_file is not None and not all_features.empty:
                dfs.append(all_features)

            # 2) selected bacteria datasets
            for b in Bacteria:
                try:
                    dfs.append(pd.read_csv(os.path.join("dataset", f"biopython-{b}.csv")))
                except FileNotFoundError as e:
                    st.error(f"Missing file for {b}: {e}")

            if dfs:
                df = (pd.concat(dfs, ignore_index=True)
                        .drop_duplicates(subset=["Sequence"])
                        .reset_index(drop=True))
                st.session_state.loaded_df = df
                st.session_state.loaded_key = cache_key
            else:
                df = None
                st.session_state.loaded_df = None
                st.session_state.loaded_key = None
        else:
            df = st.session_state.loaded_df

        data_ready = df is not None and not df.empty
        if data_ready:
            sources = []
            if upload_name:
                sources.append(f"uploaded file ({upload_name})")
            if Bacteria:
                sources.append(", ".join(Bacteria))
            st.success(f"Loaded {len(df)} sequences from: {' + '.join(sources)}")
            st.subheader("Loaded peptide data preview")
            st.dataframe(df.head())
        
        # ----------------------------
        # Run optimization
        # ----------------------------
        st.markdown("---")
        data_ready = df is not None and not df.empty
        if not data_ready:
            st.error("No peptide data available. Select bacteria or upload a valid FASTA.")
        ready = data_ready and len(opt) >= 2
        
        if st.button("🚀 Run Optimization", disabled=not ready):
            with st.spinner("Running optimization... This may take a few minutes."):
                try:
                    # algorithm setup
                    setup = algorithms_setup(
                        df=df,
                        algorithms_list=algorithms,
                        pop_size=pop_size,
                        generations=generations,
                        optimization_directions=optimization_directions,
                        length=length,
                        opt=opt,
                        constraint_dict_list=constraint_dict_list,
                        ref_targets=ref_targets,
                        seed=seed
                    )
                    setup.run_optimization()
                    setup.run()
                    st.success("Optimization completed successfully ✅")

                except Exception as e:
                    st.error(f"Error during optimization: {e}")

        # ----------------------------
        # Display cached results
        # ----------------------------
        st.subheader("📋 Optimization Results")
        st.session_state.optimization_results = st.session_state.get("optimization_results", {})
        if st.session_state.optimization_results:
            selected_algo = st.selectbox(
                "Select algorithm to view results:",
                [a for a in algorithms if a in st.session_state.optimization_results]
            )
            
            if selected_algo:
                results = st.session_state.optimization_results.get(selected_algo, None)
                if results is None:
                    st.warning(f"No results found for {selected_algo}")
                else:
                    res_dict_flipped = pd.DataFrame(results["res_dict"])
                    pareto_df_flipped = pd.DataFrame(results["pareto_df"])
                    merged_df_flipped = pd.DataFrame(results["merged_df"])
                    tab1, tab2, tab3 = st.tabs(["Objectives", "All Results", "Merged Data"])

                    with tab1:
                        st.write(f"**Objective values for {selected_algo}:**")
                        st.dataframe(res_dict_flipped)

                    with tab2:
                        st.write(f"**All optimized results for {selected_algo}:**")
                        st.dataframe(pareto_df_flipped)

                    with tab3:
                        st.write(f"**Merged data for {selected_algo}:**")
                        st.dataframe(merged_df_flipped)
        else:
            st.info("No optimization results cached yet. Run optimization first.")
            can_proceed = False

        for algo in algorithms:
            entry = st.session_state.optimization_results.get(algo)
            if entry is None:
                continue
            pareto_out = entry["pareto_df"]
            merged_out = entry["merged_df"]

            fasta_str = "".join(
                f">{row['Sequence']}\n{row['Sequence']}\n"
                for _, row in merged_out.iterrows()
            )
            cols = st.columns(3)

            #if "merged_df_flipped" not in locals() or merged_df_flipped is None:
                #st.warning(f"Skip {algo} because merged_df_flipped is not defined.")
            #    continue

            #if "pareto_df_flipped" not in locals() or pareto_df_flipped is None:
                #st.warning(f"Skip {algo} because pareto_df_flipped is not defined.")
            #    continue
            
            # -------- Local file save (for local execution) --------
            # fasta_path = os.path.join(user_home, f"{algo}.fasta")
            # with open(fasta_path, "w") as f:
            #    f.write(fasta_str)

            # pareto_df_flipped.to_csv(os.path.join(user_home, f"{algo}_all_optimize_result.csv"), index=False)
            # merged_df_flipped.to_csv(os.path.join(user_home, f"{algo}_optimize_result.csv"), index=False)
            # st.info(f"Optimize pareto front result saved at file: {user_home}\\{algo} all optimize result.csv")
            # st.info(f"Optimize pareto front result saved at file: {user_home}\\{algo} optimize result.csv")
            # st.info(f"FASTA saved at file: {fasta_path}")

            # -------- Web download --------
            with cols[0]:
                st.download_button(
                    label=f"⬇ Download {algo} FASTA",
                    data=fasta_str,
                    file_name=f"{algo}.fasta",
                    mime="text/plain",
                    key=f"download_fasta_{algo}"
                )

            with cols[1]:
                st.download_button(
                    label=f"⬇ Download {algo} all optimization results",
                    data=pareto_out.to_csv(index=False),
                    file_name=f"{algo}_all_optimize_result.csv",
                    mime="text/csv",
                    key=f"download_all_{algo}"
                )

            with cols[2]:
                st.download_button(
                    label=f"⬇ Download {algo} final optimized results",
                    data=merged_out.to_csv(index=False),
                    file_name=f"{algo}_optimize_result.csv",
                    mime="text/csv",
                    key=f"download_final_{algo}"
                )

        st.markdown("---")
        if st.button("📊 Plot Results"):
            st.session_state.show_plots = True

        if st.session_state.get("show_plots"):
            results_dict = st.session_state.get("optimization_results", {})
            if not results_dict:
                st.warning("No optimization results found. Run optimization first.")
            else:
                for algo in algorithms:
                    entry = st.session_state.optimization_results.get(algo)
                    if entry is None:
                        continue

                    plot_df = entry.get("merged_df")
                    if plot_df is None or not isinstance(plot_df, pd.DataFrame) or plot_df.empty:
                        st.warning(f"No valid DataFrame for {algo}")
                        continue

                    with st.spinner(f"Plotting results for {algo}..."):
                        st.markdown(f"### 📈 Pareto Front Visualization ({algo})")
                        amino_acid_percentage(algo, plot_df)
                        
                        directions = entry.get("directions", optimization_directions)

                        if len(directions) > 3:
                            plot_pareto_fronts_many(algo, plot_df, directions)
                        else:
                            plot_pareto_fronts_multi(algo, plot_df, directions)
                
        else:
            can_proceed = False

with main_tab2:
    st.header("System Overview")
    
    st.markdown("""
            ## 1. System Purpose
            This platform is developed for **de novo antimicrobial peptide (AMP) design and activity prediction** using machine learning models and sequence-based physicochemical representations.
    
            The system integrates:
            - sequence feature engineering
            - supervised MIC regression models
            - multi-bacteria activity profiling
            - ensemble-based prediction
    
            to support computational peptide screening against clinically relevant pathogens.
    
            ---
    
            ## 2. Computational Framework
    
            Input peptide sequences are transformed into high-dimensional descriptors, including:
            - physicochemical properties (e.g., GRAVY, charge, hydrophobicity)
            - structural propensity features
            - peptide-derived descriptors
    
            These features form the input space for predictive modeling.
            ---
    
            ## 3. Output Interpretation
    
            The system provides:
            - predicted MIC values per bacterium-model pair
            - ensemble-averaged MIC estimation
            - inferred optimal target bacteria
            - spectrum activity index (based on MIC thresholding)
            - ranking of antibacterial effectiveness
    
            ---
    
            ## 4. Intended Applications
    
            This framework is applicable to:
            - antimicrobial peptide discovery
            - in silico drug screening
            - computational microbiology research
            - sequence-function relationship studies
    
            ---
    
            ## 5. Data Source
    
            Model training is based on curated peptide-bacteria interaction datasets derived from DBAASP and related antimicrobial peptide repositories.
    
            Physicochemical features are computed using sequence-based bioinformatics descriptors.
    
            ---
    
            ## 6. Citation
    
            If you use this system, please cite:
    
            Yang C-H, Chen Y-L, Cheung T-H, Chuang L-Y.  
            *Multi-Objective Optimization Accelerates the De Novo Design of Antimicrobial Peptide for Staphylococcus aureus.*  
            International Journal of Molecular Sciences. 2024;25(24):13688.  
            https://doi.org/10.3390/ijms252413688
            """)
    
with main_tab3:
    st.header("Prediction Methodology")
    
    st.markdown("""
        ## Overview
        This module describes the computational pipeline used for antimicrobial peptide (AMP) MIC prediction.
        The framework integrates sequence-based physicochemical feature extraction with multi-model deep learning ensembles.
    
        ---
    
        ## 1. Input Representation
        Peptide sequences are provided in FASTA format and converted into structured numerical representations.
    
        Each sequence is transformed into:
    
        - Physicochemical descriptors (e.g., hydrophobicity, charge, aromaticity)
        - Structural indices (e.g., instability index, aliphatic index)
        - Peptide embedding descriptors (sequence-derived features)
    
        ---
    
        ## 2. Feature Engineering
        For each peptide sequence $S$, a feature vector $X$ is constructed:
    
        $$
        X = [f_1(S), f_2(S), ..., f_n(S)]
        $$
    
        where features include:
    
        - Hydrophobicity (GRAVY)
        - Isoelectric point
        - Net charge
        - Molecular weight
        - Secondary structure fractions
        - Boman index
        - Autocorrelation & covariance descriptors
        - Amino acid composition-based descriptors
        - Charge at pH
    
        All features are normalized using **RobustScaler** to reduce sensitivity to outliers.
    
        ---
    
        ## 3. Bacteria-aware Encoding
        The model incorporates bacterial conditioning via label encoding:
    
        $$
        y = f(X, b)
        $$
    
        where:
        - $X$: peptide features
        - $b$: bacterial species index (0–3)
    
        This allows species-specific MIC prediction.
    
        ---
    
        ## 4. Model Architecture
        An ensemble of deep learning models is used:
    
        - CNN-based regression model
        - AutoEncoder regressor (AE)
        - Denoising AutoEncoder (DAE)
        - Variational AutoEncoder (VAE)
        - Contrastive representation model
    
        Each model outputs:
    
        $$
        \hat{y}_{log2MIC}
        $$
    
        which is converted to MIC scale:
    
        $$
        MIC = 2^{\hat{y}} - 1
        $$
    
        ---
    
        ## 5. Ensemble Strategy
        For each bacteria species:
    
        $$
        MIC_{ensemble} = \frac{1}{M} \sum_{i=1}^{M} MIC_i
        $$
    
        where $M$ is the number of models.
    
        ---
    
        ## 6. Decision Logic
        The system provides:
    
        - Best target bacteria (minimum MIC)
        - Broad-spectrum score (MIC < threshold)
        - Ranked antibacterial susceptibility profile
    
        ---
    
        ## 7. Output Interpretation
        Predictions represent **in silico estimated MIC values** and should be interpreted as:
    
        > computational guidance for peptide prioritization, not experimental validation.
    
        ---
        """)
    
with main_tab4:
    st.header("How to Design Antimicrobial Peptides using this App")
    st.markdown("""
        This app integrates multi-objective optimization frameworks with peptide physicochemical profiling.  
        Follow the workflow below to construct customized antimicrobial peptide (AMP) candidates.
    
        ### 1. **Select Target Bacteria**
        Choose one or more pathogens from the sidebar.  
        The app will automatically load precomputed physicochemical property datasets associated with the selected species.
        Or you can upload your own peptide sequences in .fasta or .fa format, and the app will compute their properties for optimization.
        *Note*: Upload files should contain *Standard amino acids peptide* sequences. FASTA files must have proper headers, while TXT files should list one sequence per line.
    
        ### 2. **Choose Optimization Algorithms**
        Select one or multiple multi-objective evolutionary algorithms (MOEAs), such as:
        - NSGA-II / NSGA-III  
        - R-NSGA-II / R-NSGA-III  
        - U-NSGA-III  
        - AGE-MOEA / AGE-MOEA-II  
    
        Each algorithm presents different strengths in balancing exploration and convergence toward high-quality Pareto-optimal peptides.
    
        ### 3. **Define Optimization Parameters**
        - **Population Size**: Controls the diversity of candidate sequences within the evolutionary search.
        - **Peptide Length**: Specifies the length of generated sequences for de novo design.
        - **Generations**: Determines the number of iterations for algorithmic evolution.
    
        These parameters directly influence search convergence, diversity maintenance, and computational runtime.
                    
        """)
    st.image(os.path.join("figure", "side bar.png"))
    st.markdown("""
        ### 4. **Select Physicochemical Objectives**
        Choose at least two descriptors for optimization.  
        Available objectives include:
        - Gravy(Hydrophobicity)
        - Instability Index  
        - Isoelectric Point  
        - Net Charge  
        - Aliphatic Index  
        - Aromaticity  
        - Molecular Weight  
        - Boman Index  
        - Secondary Structure Fractions (Helix, Turn, Sheet)
    
        For each objective, specify whether the algorithm should **minimize** or **maximize** the descriptor.  
        (Gravy is treated as *hydrophilicity* or *hydrophobicity* depending on user preference.)
                    """)
    st.image(os.path.join("figure", "Objectives to optimize.png"))
    st.markdown("""
    ### 5. **Add Optional Constraints**
    Users may define upper/lower bounds to restrict the peptide search space.  
    For example:
    - Instability Index ≥ 40  
    - 1 ≤ Net Charge
    - Gravy ≥ 1

    Constraints help enforce biologically realistic design regions and improve hit quality.
                """)
    st.image(os.path.join("figure", "constraints.png"))
    st.markdown("""
    ### 6. **Run Optimization**
    Press **“Run Optimization”** to execute the selected algorithms.  
    The app will:
    - Perform evolutionary optimization  
    - Compute Pareto fronts  
    - Identify non-dominated peptide solutions  
    - Cache results for further analysis  

    ### 7. **View and Download Results**
    Results include:
    - Objective value tables
    - Full Pareto-optimal peptide lists
    - Merged physicochemical property profiles
    - FASTA files for external analysis  
    - CSV exports for downstream modeling

    ### 8. **Visualize Optimization Outcomes**
    You may generate:
    - Multi-dimensional Pareto front plots  
    - Amino acid composition heatmaps  
    - Sequence-level physicochemical distribution analyses  

    These visualizations provide insight into peptide behavior, trade-offs among descriptors, and optimization dynamics.
                
                """)
    st.image(os.path.join("figure", "run.png"))
    st.image(os.path.join("figure", "result.png"))
    st.image(os.path.join("figure", "plot.png"))
    st.markdown("""
    ### Summary
    This app provides a structured, multi-objective approach to AMP design by integrating algorithmic search, physicochemical evaluation, and biological constraint modeling. It aims to accelerate the rational development of antimicrobial peptides with optimized properties.
    If you encounter any issues or have questions, please upload your issues or figure to the 
                https://github.com/ksuee108/amp_desige/issues, and we will get back to you as soon as possible.
    """)
with main_tab5:
    st.header("Related Databases and Prediction Websites")
    AMP_databases = {
        "Website":["Peptaibols", "Cybase", "BACTIBASE", "CAMP", "HIPdb", "Hemolytik", "ParaPep", "CancerPPD/AntiCP 2.0", "DBAASP", "BaAMPs", "SATPdb", "DRAMP", "InverPep", "MBPDB", "AntiTbPdb", "LABiocin", "ADAPTABLE", "FoldamerDB", "AntiCP 2.0", "FermFooDb", "B-AMP", "SuPepMem", "ACovPepDB", "AMPDB v1", "DRAVP", "GtoPdb", "aSynPEP-DB", "AbAMPdb", "AVR/I/SSAPDB", "TAMRSA", "IAMPDB", "ABPDB"],
        "Link":["https://peptaibol.cryst.bbk.ac.uk/home.shtml", "https://www.cybase.org.au/", "https://bactibase.pfba-lab-tun.org/main.php", "http://www.bicnirrh.res.in/antimicrobial", "http://crdd.osdd.net/servers/hipdb/", "http://crdd.osdd.net/raghava/hemolytik/", "http://crdd.osdd.net/raghava/parapep/", "http://crdd.osdd.net/raghava/cancerppd/", "http://dbaasp.org/home.xhtml", "http://www.baamps.it/", "http://crdd.osdd.net/raghava/satpdb/", "http://dramp.cpu-bioinfor.org/", "http://ciencias.medellin.unal.edu.co/gruposdeinvestigacion/prospeccionydisenobiomoleculas/InverPep/public/home_en", "https://mbpdb.nws.oregonstate.edu/", "http://webs.iiitd.edu.in/raghava/antitbpdb/", "https://labiocin.univ-lille.fr/", "http://gec.u-picardie.fr/adaptable", "http://foldamerdb.ttk.hu/", "https://webs.iiitd.edu.in/raghava/anticp2/", "https://webs.iiitd.edu.in/raghava/fermfoodb/", "https://b-amp.karishmakaushiklab.com/", "https://supepmem.com/", "http://i.uestc.edu.cn/ACovPepDB/", "https://bblserver.org.in/ampdb/", "http://dravp.cpu-bioinfor.org/", "https://www.guidetopharmacology.org", "https://asynpepdb.ppmclab.com/", "https://abampdb.mgbio.tech/", "https://bblserver.org.in/avrissa/", "https://bblserver.org.in/tamrsar/", "https://bblserver.org.in/iampdb/", "http://www.acdb.plus/ABPDB"],
    }

    st.markdown("### 1. AMP Databases")
    df_AMP_databases = pd.DataFrame(AMP_databases)
    st.table(df_AMP_databases)

    st.markdown("### 2. AMP Prediction Websites")
    AMP_prediction_websites = {
        "Website":["BAGLE", "AntiBP", "AMPer", "CAMP Prediction", "antiSMASH", "AMPA", "AMP_Scanner", "DBAASP", "AI4AXP"],
        "Link":["http://bagel.molgenrug.nl/", "https://webs.iiitd.edu.in/raghava/antibp/submit.html", "http://marray.cmdr.ubc.ca/cgi-bin/amp.pl", "http://www.camp.bicnirrh.res.in/predict/", "http://antismash.secondarymetabolites.org/", "http://tcoffee.crg.cat/apps/ampa/do", "http://www.ampscanner.com", "http://dbaasp.org/home.xhtml", "https://axp.iis.sinica.edu.tw/"]
    }
    df_AMP_prediction_websites = pd.DataFrame(AMP_prediction_websites)
    st.table(df_AMP_prediction_websites)