import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import json
import io
import base64
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass, asdict
import uuid


# Data Models
@dataclass
class Role:
    id: str
    name: str
    commission_formula: str
    default_monthly_quota: float
    default_target_incentive: float


@dataclass
class SalesRep:
    id: str
    name: str
    role_id: str
    manager_id: Optional[str]
    email: str
    monthly_settings: Dict[str, Dict[str, float]]


@dataclass
class Sale:
    id: str
    rep_id: Optional[str]
    rep_name: str
    sale_amount: float
    product: str
    date: str
    customer: str
    commission: float = 0.0


class CommissionCalculator:
    @staticmethod
    def calculate_commission(sale_amount: float, total_sales: float, formula: str) -> float:
        """Safely evaluate commission formula"""
        try:
            # Create a safe environment for formula evaluation
            safe_dict = {
                "saleAmount": sale_amount,
                "totalSales": total_sales,
                "__builtins__": {}
            }
            result = eval(formula, safe_dict)
            return float(result) if result is not None else 0.0
        except Exception as e:
            print(f"Commission calculation error: {e}") #For debugging
            return 0.0


class CommissionApp:
    def __init__(self):
        self.init_session_state()

    def init_session_state(self):
        """Initialize session state with default data"""
        if 'roles' not in st.session_state:
            st.session_state.roles = [
                Role(
                    id="1",
                    name="Account Executive",
                    commission_formula="saleAmount * 0.10 if saleAmount >= 50000 else saleAmount * 0.07 if saleAmount >= 25000 else saleAmount * 0.05 if saleAmount >= 10000 else saleAmount * 0.03",
                    default_monthly_quota=50000,
                    default_target_incentive=5000
                ),
                Role(
                    id="2",
                    name="Sales Manager",
                    commission_formula="saleAmount * 0.06 if saleAmount >= 75000 else saleAmount * 0.04 if saleAmount >= 25000 else saleAmount * 0.02",
                    default_monthly_quota=150000,
                    default_target_incentive=10000
                )
            ]

        if 'sales_reps' not in st.session_state:
            st.session_state.sales_reps = [
                SalesRep(
                    id="1",
                    name="John Smith",
                    role_id="1",
                    manager_id=None,
                    email="john.smith@company.com",
                    monthly_settings={}
                ),
                SalesRep(
                    id="2",
                    name="Jane Doe",
                    role_id="1",
                    manager_id="3",
                    email="jane.doe@company.com",
                    monthly_settings={}
                ),
                SalesRep(
                    id="3",
                    name="Mike Johnson",
                    role_id="2",
                    manager_id=None,
                    email="mike.johnson@company.com",
                    monthly_settings={}
                )
            ]

        if 'sales_data' not in st.session_state:
            st.session_state.sales_data = []

        if 'selected_month' not in st.session_state:
            st.session_state.selected_month = datetime.now().strftime("%Y-%m")

        # Add data update tracking
        if 'data_last_updated' not in st.session_state:
            st.session_state.data_last_updated = datetime.now()

        if 'commission_last_calculated' not in st.session_state:
            st.session_state.commission_last_calculated = None

    def get_role_by_id(self, role_id: str) -> Optional[Role]:
        """Get role by ID"""
        return next((role for role in st.session_state.roles if role.id == role_id), None)

    def get_rep_by_id(self, rep_id: str) -> Optional[SalesRep]:
        """Get sales rep by ID"""
        return next((rep for rep in st.session_state.sales_reps if rep.id == rep_id), None)

    def get_monthly_data(self, month: str) -> List[Sale]:
        """Get sales data for specific month"""
        return [sale for sale in st.session_state.sales_data if sale.date.startswith(month)]

    def get_rep_monthly_settings(self, rep: SalesRep, month: str) -> Dict[str, float]:
        """Get rep's monthly settings with defaults"""
        month_settings = rep.monthly_settings.get(month, {})
        role = self.get_role_by_id(rep.role_id)
        return {
            'quota': month_settings.get('quota', role.default_monthly_quota if role else 0),
            'target_incentive': month_settings.get('target_incentive', role.default_target_incentive if role else 0)
        }

    def calculate_commissions(self):
        """Calculate commissions for all sales - CENTRALIZED CALCULATION"""
        updated_sales = []

        for sale in st.session_state.sales_data:
            rep = self.get_rep_by_id(sale.rep_id) if sale.rep_id else None
            role = self.get_role_by_id(rep.role_id) if rep else None

            if not role or not role.commission_formula:
                sale.commission = 0
            else:
                # Calculate total sales for the rep across all time periods
                total_sales = sum(s.sale_amount for s in st.session_state.sales_data if s.rep_id == sale.rep_id)
                sale.commission = CommissionCalculator.calculate_commission(
                    sale.sale_amount, total_sales, role.commission_formula
                )

            updated_sales.append(sale)

        st.session_state.sales_data = updated_sales
        st.session_state.commission_last_calculated = datetime.now()

        # Force update to notify all tabs
        st.session_state.data_last_updated = datetime.now()

    def get_rep_summary(self, month: str) -> List[Dict]:
        """Get summary data for all reps for specific month - USES CENTRALIZED DATA"""
        monthly_data = self.get_monthly_data(month)
        rep_map = {}

        # Process sales data from centralized dataset
        for sale in monthly_data:
            rep = self.get_rep_by_id(sale.rep_id) if sale.rep_id else None
            if not rep:
                continue

            if sale.rep_id not in rep_map:
                role = self.get_role_by_id(rep.role_id)
                month_settings = self.get_rep_monthly_settings(rep, month)
                rep_map[sale.rep_id] = {
                    'id': rep.id,
                    'name': rep.name,
                    'role': role.name if role else 'No Role',
                    'email': rep.email,
                    'manager_id': rep.manager_id,
                    'total_sales': 0,
                    'total_commission': 0,
                    'deals': 0,
                    'quota': month_settings['quota'],
                    'target_incentive': month_settings['target_incentive']
                }

            rep_map[sale.rep_id]['total_sales'] += sale.sale_amount
            rep_map[sale.rep_id]['total_commission'] += sale.commission
            rep_map[sale.rep_id]['deals'] += 1

        # Add reps with no sales
        for rep in st.session_state.sales_reps:
            if rep.id not in rep_map:
                role = self.get_role_by_id(rep.role_id)
                month_settings = self.get_rep_monthly_settings(rep, month)
                rep_map[rep.id] = {
                    'id': rep.id,
                    'name': rep.name,
                    'role': role.name if role else 'No Role',
                    'email': rep.email,
                    'manager_id': rep.manager_id,
                    'total_sales': 0,
                    'total_commission': 0,
                    'deals': 0,
                    'quota': month_settings['quota'],
                    'target_incentive': month_settings['target_incentive']
                }

        return list(rep_map.values())

    def render_header(self):
        """Render application header with REAL-TIME DATA"""
        st.set_page_config(page_title="Commission Platform", layout="wide")

        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

        with col1:
            st.title("🧮 Appgate Commission Platform")

        with col2:
            # Month selector that affects all tabs
            new_month = st.date_input(
                "Select Month",
                value=datetime.strptime(st.session_state.selected_month + "-01", "%Y-%m-%d").date(),
                format="YYYY-MM-DD"
            ).strftime("%Y-%m")

            # Update month if changed
            if new_month != st.session_state.selected_month:
                st.session_state.selected_month = new_month
                st.rerun()

        with col3:
            # Real-time metrics from centralized data
            monthly_data = self.get_monthly_data(st.session_state.selected_month)
            total_sales = sum(sale.sale_amount for sale in monthly_data)
            total_commissions = sum(sale.commission for sale in monthly_data)
            st.metric("Monthly Sales", f"${total_sales:,.0f}")

        with col4:
            st.metric("Monthly Commissions", f"${total_commissions:,.0f}")

        # Data status indicator
        if st.session_state.sales_data:
            calculated_count = len([sale for sale in st.session_state.sales_data if sale.commission > 0])
            total_count = len(st.session_state.sales_data)

            if calculated_count == total_count and calculated_count > 0:
                st.success(f"✅ All {total_count} sales records have calculated commissions")
            elif calculated_count > 0:
                st.warning(f"⚠️ {calculated_count}/{total_count} sales records have calculated commissions")
            else:
                st.error(f"❌ No commissions calculated for {total_count} sales records")
        else:
            st.info("📤 No sales data loaded. Use the Upload Data tab to get started.")

    def render_dashboard(self):
        """Render dashboard tab - USES CENTRALIZED DATA"""
        monthly_data = self.get_monthly_data(st.session_state.selected_month)
        total_sales = sum(sale.sale_amount for sale in monthly_data)
        total_commissions = sum(sale.commission for sale in monthly_data)
        rep_summary = self.get_rep_summary(st.session_state.selected_month)

        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Monthly Sales", f"${total_sales:,.0f}")
        with col2:
            st.metric("Monthly Commissions", f"${total_commissions:,.0f}")
        with col3:
            st.metric("Active Reps", len([rep for rep in rep_summary if rep['total_sales'] > 0]))
        with col4:
            st.metric("Monthly Deals", len(monthly_data))

        # Show data source info
        if st.session_state.sales_data:
            st.info(f"📊 Displaying data from {len(st.session_state.sales_data)} total sales records")

            if st.session_state.commission_last_calculated:
                calc_time = st.session_state.commission_last_calculated.strftime("%Y-%m-%d %H:%M:%S")
                st.caption(f"Last commission calculation: {calc_time}")

        # Rep Performance Table
        st.subheader(f"Rep Performance - {st.session_state.selected_month}")

        if rep_summary:
            df = pd.DataFrame(rep_summary)
            df['attainment'] = (df['total_sales'] / df['quota'] * 100).round(1)
            df['attainment_color'] = df['attainment'].apply(
                lambda x: '🟢' if x >= 100 else '🟡' if x >= 80 else '🔴'
            )

            display_df = df[
                ['attainment_color', 'name', 'total_sales', 'quota', 'attainment', 'total_commission']].copy()
            display_df.columns = ['Status', 'Rep', 'Sales', 'Quota', 'Attainment %', 'Commission']
            display_df['Sales'] = display_df['Sales'].apply(lambda x: f"${x:,.0f}")
            display_df['Quota'] = display_df['Quota'].apply(lambda x: f"${x:,.0f}")
            display_df['Commission'] = display_df['Commission'].apply(lambda x: f"${x:,.0f}")

            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No rep data available for the selected month.")

    def render_upload_data(self):
        """Render upload data tab - PRIMARY DATA SOURCE"""
        st.subheader("Upload Sales Data")

        st.info("Expected CSV columns: rep_name, sale_amount, product, date, customer")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                # Validate columns
                required_columns = ['rep_name', 'sale_amount', 'product', 'date', 'customer']
                if not all(col in df.columns for col in required_columns):
                    st.error(f"CSV must contain columns: {', '.join(required_columns)}")
                    return

                # Process data
                processed_sales = []
                for _, row in df.iterrows():
                    rep = next((r for r in st.session_state.sales_reps
                                if r.name.lower() == str(row['rep_name']).lower().strip()), None)

                    sale = Sale(
                        id=str(uuid.uuid4()),
                        rep_id=rep.id if rep else None, #Matching sales order to rep
                        rep_name=str(row['rep_name']).strip(),
                        sale_amount=float(row['sale_amount']) if pd.notna(row['sale_amount']) else 0,
                        product=str(row['product']).strip(),
                        date=str(row['date']), #Date for monthly grouping
                        customer=str(row['customer']).strip()
                    )

                    if sale.rep_name and sale.sale_amount > 0:
                        processed_sales.append(sale)

                # After processing uploaded sales
                unmatched_reps = []
                for sale in processed_sales:
                    if not sale.rep_id:
                        unmatched_reps.append(sale.rep_name)

                if unmatched_reps:
                    st.warning(f"These reps couldn't be matched: {set(unmatched_reps)}")

                # Option to append or replace existing data
                col1, col2 = st.columns(2)
                with col1:
                    data_action = st.radio(
                        "Data Action:",
                        ["Replace existing data", "Append to existing data"],
                        key="data_action"
                    )

                with col2:
                    auto_calculate = st.checkbox(
                        "Auto-calculate commissions after upload",
                        value=True,
                        help="Automatically calculate commissions when new data is uploaded"
                    )

                if st.button("Upload Data", type="primary"):
                    if data_action == "Replace existing data":
                        st.session_state.sales_data = processed_sales
                        st.success(
                            f"Successfully uploaded {len(processed_sales)} sales records (replaced existing data)")
                    else:
                        # Append to existing data
                        existing_data = st.session_state.sales_data.copy()
                        existing_data.extend(processed_sales)
                        st.session_state.sales_data = existing_data
                        st.success(f"Successfully appended {len(processed_sales)} sales records to existing data")

                    # Update timestamp to notify other tabs
                    st.session_state.data_last_updated = datetime.now()

                    if auto_calculate:
                        with st.spinner("Calculating commissions..."):
                            self.calculate_commissions()
                            st.success("Commissions calculated successfully!")
                    else:
                        st.warning("Commissions not calculated. Use the 'Calculate Commissions' button below.")

                    st.rerun()

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    def render_data_summary (self):
        # Commission Calculation Controls
        st.divider()
        st.subheader("Commission Calculation")

        if st.session_state.sales_data:
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("🧮 Calculate All Commissions", type="primary"):
                    with st.spinner("Calculating commissions for all sales..."):
                        self.calculate_commissions()
                        st.success("All commissions calculated successfully! Changes will appear in all tabs.")
                        st.rerun()

            with col2:
                if st.button("🔄 Recalculate Commissions"):
                    with st.spinner("Recalculating commissions..."):
                        # Reset all commissions to 0 first
                        for sale in st.session_state.sales_data:
                            sale.commission = 0.0
                        self.calculate_commissions()
                        st.success("All commissions recalculated! Changes reflected across all tabs.")
                        st.rerun()

            with col3:
                if st.button("🗑️ Clear All Data", type="secondary"):
                    if st.session_state.get('confirm_clear', False):
                        st.session_state.sales_data = []
                        st.session_state.data_last_updated = datetime.now()
                        st.session_state.commission_last_calculated = None
                        st.session_state.confirm_clear = False
                        st.success("All sales data cleared! All tabs updated.")
                        st.rerun()
                    else:
                        st.session_state.confirm_clear = True
                        st.warning("Click again to confirm data deletion")

            # Show calculation status
            calculated_sales = [sale for sale in st.session_state.sales_data if sale.commission > 0]
            total_sales = len(st.session_state.sales_data)

            if calculated_sales:
                st.info(f"✅ Commissions calculated for {len(calculated_sales)} out of {total_sales} sales records")
            else:
                st.warning(f"⚠️ No commissions calculated yet for {total_sales} sales records")
        else:
            st.info("Upload sales data to enable commission calculations")

        # Historical Data Section
        st.subheader("Historical Sales Data")

        if st.session_state.sales_data:
            # Create DataFrame from centralized sales data
            historical_data = []
            for sale in st.session_state.sales_data:
                # Get rep role for additional context
                rep = self.get_rep_by_id(sale.rep_id) if sale.rep_id else None
                role = self.get_role_by_id(rep.role_id) if rep else None

                historical_data.append({
                    'Date': sale.date,
                    'Rep Name': sale.rep_name,
                    'Role': role.name if role else 'Unknown',
                    'Product': sale.product,
                    'Customer': sale.customer,
                    'Sale Amount': sale.sale_amount,
                    'Commission': sale.commission
                })

            historical_df = pd.DataFrame(historical_data)

            # Sort by date (most recent first)
            historical_df = historical_df.sort_values('Date', ascending=False)

            # Add commission calculation status indicators
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                total_sales = historical_df['Sale Amount'].sum()
                st.metric("Total Sales", f"${total_sales:,.0f}")

            with col2:
                total_commissions = historical_df['Commission'].sum()
                st.metric("Total Commissions", f"${total_commissions:,.2f}")

            with col3:
                total_deals = len(historical_df)
                st.metric("Total Deals", total_deals)

            with col4:
                avg_deal_size = historical_df['Sale Amount'].mean()
                st.metric("Avg Deal Size", f"${avg_deal_size:,.0f}")

            with col5:
                calculated_count = len([sale for sale in st.session_state.sales_data if sale.commission > 0])
                commission_rate = (calculated_count / len(
                    st.session_state.sales_data)) * 100 if st.session_state.sales_data else 0
                st.metric("Commissions Calculated", f"{commission_rate:.1f}%")

            # Add filters
            st.subheader("Filter Options")
            col1, col2, col3 = st.columns(3)

            with col1:
                # Rep filter
                rep_names = ['All'] + sorted(historical_df['Rep Name'].unique().tolist())
                selected_rep = st.selectbox("Filter by Rep:", rep_names)

            with col2:
                # Product filter
                products = ['All'] + sorted(historical_df['Product'].unique().tolist())
                selected_product = st.selectbox("Filter by Product:", products)

            with col3:
                # Date range filter
                date_range = st.date_input(
                    "Filter by Date Range:",
                    value=[],
                    help="Leave empty to show all dates"
                )

            # Apply filters
            filtered_df = historical_df.copy()

            if selected_rep != 'All':
                filtered_df = filtered_df[filtered_df['Rep Name'] == selected_rep]

            if selected_product != 'All':
                filtered_df = filtered_df[filtered_df['Product'] == selected_product]

            if date_range and len(date_range) == 2:
                start_date, end_date = date_range
                filtered_df = filtered_df[
                    (pd.to_datetime(filtered_df['Date']).dt.date >= start_date) &
                    (pd.to_datetime(filtered_df['Date']).dt.date <= end_date)
                    ]

            # Display filtered data with commission status indicators
            st.subheader(f"Sales Records ({len(filtered_df)} records)")

            # Format display DataFrame with status indicators
            display_df = filtered_df.copy()
            display_df['Commission Status'] = display_df['Commission'].apply(
                lambda x: '✅ Calculated' if x > 0 else '⏳ Pending'
            )
            display_df['Sale Amount'] = display_df['Sale Amount'].apply(lambda x: f"${x:,.0f}")
            display_df['Commission'] = display_df['Commission'].apply(lambda x: f"${x:,.2f}")

            # Reorder columns to show status first
            column_order = ['Commission Status', 'Date', 'Rep Name', 'Role', 'Product', 'Customer', 'Sale Amount',
                            'Commission']
            display_df = display_df[column_order]

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Export option
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("📊 Export Filtered Data"):
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"historical_sales_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        else:
            st.info("No historical sales data available. Upload a CSV file to see historical data here.")

    def render_roles(self):
        """Render roles and rules tab - AFFECTS COMMISSION CALCULATIONS"""
        st.subheader("Roles & Commission Rules")

        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("➕ Create Role", type="primary"):
                st.session_state.show_role_modal = True

        # Show impact on existing data
        if st.session_state.sales_data:
            st.info(
                f"💡 Changes to roles will affect commission calculations for {len(st.session_state.sales_data)} existing sales records")

        # Display roles
        for i, role in enumerate(st.session_state.roles):
            with st.expander(f"📋 {role.name}", expanded=False):
                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    st.metric("Default Monthly Quota", f"${role.default_monthly_quota:,.0f}")

                with col2:
                    st.metric("Default Target Incentive", f"${role.default_target_incentive:,.0f}")

                with col3:
                    if st.button("✏️ Edit", key=f"edit_role_{i}"):
                        st.session_state.editing_role = role
                        st.session_state.show_role_modal = True

                st.text("Commission Formula:")
                st.code(role.commission_formula, language="python")

                # Show which reps use this role
                reps_with_role = [rep for rep in st.session_state.sales_reps if rep.role_id == role.id]
                if reps_with_role:
                    st.caption(f"Used by: {', '.join([rep.name for rep in reps_with_role])}")

        # Role modal
        if st.session_state.get('show_role_modal', False):
            self.render_role_modal()

    def render_role_modal(self):
        """Render role creation/editing modal"""
        with st.form("role_form"):
            st.subheader("Create/Edit Role")

            editing_role = st.session_state.get('editing_role')

            name = st.text_input("Role Name", value=editing_role.name if editing_role else "")
            default_quota = st.number_input("Default Monthly Quota",
                                            value=editing_role.default_monthly_quota if editing_role else 50000,
                                            step=1000)
            default_incentive = st.number_input("Default Target Incentive",
                                                value=editing_role.default_target_incentive if editing_role else 5000,
                                                step=100)

            st.text("Commission Formula:")
            st.caption("Available variables: saleAmount, totalSales")
            formula = st.text_area(
                "Formula",
                value=editing_role.commission_formula if editing_role else "saleAmount * 0.05 if saleAmount >= 10000 else saleAmount * 0.03",
                height=100
            )

            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                if st.form_submit_button("Test Formula"):
                    try:
                        test_commission = CommissionCalculator.calculate_commission(10000, 10000, formula)
                        st.success(f"Formula test with $10,000 sale: ${test_commission:,.2f} commission")
                    except Exception as e:
                        st.error(f"Formula error: {str(e)}")

            with col2:
                if st.form_submit_button("Save Role", type="primary"):
                    if editing_role:
                        # Update existing role
                        for i, role in enumerate(st.session_state.roles):
                            if role.id == editing_role.id:
                                st.session_state.roles[i] = Role(
                                    id=editing_role.id,
                                    name=name,
                                    commission_formula=formula,
                                    default_monthly_quota=default_quota,
                                    default_target_incentive=default_incentive
                                )
                    else:
                        # Create new role
                        new_role = Role(
                            id=str(uuid.uuid4()),
                            name=name,
                            commission_formula=formula,
                            default_monthly_quota=default_quota,
                            default_target_incentive=default_incentive
                        )
                        st.session_state.roles.append(new_role)

                    st.session_state.show_role_modal = False
                    if 'editing_role' in st.session_state:
                        del st.session_state.editing_role

                    # Trigger recalculation warning
                    if st.session_state.sales_data:
                        st.warning("⚠️ Role updated! Go to Upload Data tab to recalculate commissions.")

                    st.success("Role saved successfully!")
                    st.rerun()

            with col3:
                if st.form_submit_button("Cancel"):
                    st.session_state.show_role_modal = False
                    if 'editing_role' in st.session_state:
                        del st.session_state.editing_role
                    st.rerun()

    def render_sales_reps(self):
        """Render sales reps tab with inline editable table view organized by role"""
        st.subheader("Sales Representatives Management")

        # Action buttons at the top
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            # Show connection to sales data
            if st.session_state.sales_data:
                active_reps = len(set(sale.rep_id for sale in st.session_state.sales_data if sale.rep_id))
                st.info(f"{active_reps} reps have sales data in the system")

        with col2:
            if st.button("Add New Rep", type="primary", key="add_rep_btn"):
                st.session_state.show_rep_modal = True
                st.session_state.editing_rep = None

        with col3:
            if st.button("Clear All Reps", type="secondary", key="clear_all_reps"):
                if st.session_state.get('confirm_clear_reps', False):
                    st.session_state.sales_reps = []
                    st.session_state.confirm_clear_reps = False
                    st.success("All sales representatives cleared!")
                    st.rerun()
                else:
                    st.session_state.confirm_clear_reps = True
                    st.warning("Click again to confirm deletion of all reps")

        # Initialize editing state
        if 'editing_field' not in st.session_state:
            st.session_state.editing_field = None

        # Get rep summary data
        rep_summary = self.get_rep_summary(st.session_state.selected_month)

        if not rep_summary:
            st.info("No sales representatives found. Add some reps to get started!")
            if st.session_state.get('show_rep_modal', False):
                self.render_rep_modal()
            return

        # Create DataFrame and organize by role
        df = pd.DataFrame(rep_summary)
        df['attainment'] = (df['total_sales'] / df['quota'] * 100).round(1)

        # Add manager names and status
        df['manager_name'] = df['manager_id'].apply(
            lambda x: next((rep.name for rep in st.session_state.sales_reps if rep.id == x),
                           'None') if x else 'None'
        )

        df['status_icon'] = df['attainment'].apply(
            lambda x: '🟢' if x >= 100 else '🟡' if x >= 80 else '🔴'
        )

        # Group by role and create editable tables
        roles = sorted(df['role'].unique())

        for role in roles:
            role_df = df[df['role'] == role].sort_values('name')

            st.subheader(f"{role} ({len(role_df)} representatives)")

            # Create table header with expanded columns
            header_cols = st.columns([0.4, 1.0, 1.3, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.6])
            headers = ['Status', 'Rep ID', 'Name', 'Email', 'Manager', 'Monthly Sales', 'Commission', 'Quota',
                       'Target Incentive', 'Attainment', 'Deals', 'Save']

            for i, (col, header) in enumerate(zip(header_cols, headers)):
                with col:
                    st.write(f"**{header}**")

            # Create editable rows for this role
            for idx, (_, rep_data) in enumerate(role_df.iterrows()):
                rep_obj = next((rep for rep in st.session_state.sales_reps if rep.id == rep_data['id']), None)

                row_cols = st.columns([0.4, 1.0, 1.3, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.6])

                # Status (non-editable)
                with row_cols[0]:
                    st.write(rep_data['status_icon'])

                # Rep ID (editable)
                with row_cols[1]:
                    current_rep_id = rep_obj.id if rep_obj else rep_data['id']
                    new_rep_id = st.text_input(
                        label="rep_id",
                        value=current_rep_id,
                        key=f"rep_id_{rep_data['id']}",
                        label_visibility="collapsed",
                        help="Edit rep ID"
                    )

                # Name (editable)
                with row_cols[2]:
                    current_name = rep_obj.name if rep_obj else rep_data['name']
                    new_name = st.text_input(
                        label="name",
                        value=current_name,
                        key=f"name_{rep_data['id']}",
                        label_visibility="collapsed",
                        help="Edit name"
                    )

                # Email (editable)
                with row_cols[3]:
                    current_email = rep_obj.email if rep_obj else rep_data['email']
                    new_email = st.text_input(
                        label="email",
                        value=current_email,
                        key=f"email_{rep_data['id']}",
                        label_visibility="collapsed",
                        help="Edit email"
                    )

                # Manager (editable dropdown)
                with row_cols[4]:
                    manager_options = [("", "None")] + [(rep.id, rep.name) for rep in st.session_state.sales_reps if
                                                        rep.id != rep_data['id']]
                    manager_names = [name for _, name in manager_options]

                    current_manager_idx = 0
                    if rep_obj and rep_obj.manager_id:
                        try:
                            current_manager_idx = next(
                                i for i, (mgr_id, _) in enumerate(manager_options) if mgr_id == rep_obj.manager_id)
                        except StopIteration:
                            current_manager_idx = 0

                    new_manager_name = st.selectbox(
                        label="manager",
                        options=manager_names,
                        index=current_manager_idx,
                        key=f"manager_{rep_data['id']}",
                        label_visibility="collapsed",
                        help="Select manager"
                    )

                # Monthly Sales (display only)
                with row_cols[5]:
                    st.write(f"${rep_data['total_sales']:,.0f}")

                # Commission (display only)
                with row_cols[6]:
                    st.write(f"${rep_data['total_commission']:,.2f}")

                # Quota (editable with currency formatting)
                with row_cols[7]:
                    current_quota = rep_data['quota']
                    quota_display = f"${current_quota:,.0f}"
                    new_quota_text = st.text_input(
                        label="quota",
                        value=quota_display,
                        key=f"quota_{rep_data['id']}",
                        label_visibility="collapsed",
                        help="Edit monthly quota (e.g., $50,000)",
                    )
                    # Parse the quota value, removing $ and commas
                    try:
                        cleaned_quota = new_quota_text.replace('$', '').replace(',', '').strip()
                        new_quota = float(cleaned_quota) if cleaned_quota else current_quota
                    except (ValueError, AttributeError):
                        new_quota = current_quota

                # Target Incentive (editable with currency formatting)
                with row_cols[8]:
                    current_incentive = rep_data['target_incentive']
                    incentive_display = f"${current_incentive:,.0f}"
                    new_incentive_text = st.text_input(
                        label="target_incentive",
                        value=incentive_display,
                        key=f"incentive_{rep_data['id']}",
                        label_visibility="collapsed",
                        help="Edit target incentive (e.g., $5,000)",
                    )
                    # Parse the incentive value, removing $ and commas
                    try:
                        cleaned_incentive = new_incentive_text.replace('$', '').replace(',', '').strip()
                        new_incentive = float(cleaned_incentive) if cleaned_incentive else current_incentive
                    except (ValueError, AttributeError):
                        new_incentive = current_incentive

                # Attainment (display only)
                with row_cols[9]:
                    st.write(f"{rep_data['attainment']:.1f}%")

                # Deals (display only)
                with row_cols[10]:
                    st.write(str(rep_data['deals']))

                # Save button
                with row_cols[11]:
                    if st.button("💾", key=f"save_rep_{rep_data['id']}", help=f"Save changes for {rep_data['name']}"):
                        # Validate Rep ID uniqueness
                        existing_ids = [rep.id for rep in st.session_state.sales_reps if rep.id != rep_data['id']]
                        if new_rep_id in existing_ids:
                            st.error(f"Rep ID '{new_rep_id}' already exists!")
                            continue

                        # Get selected manager ID
                        selected_manager_id = None
                        if new_manager_name != "None":
                            selected_manager_id = next(
                                mgr_id for mgr_id, mgr_name in manager_options if mgr_name == new_manager_name)

                        # Update the rep object
                        if rep_obj:
                            for i, rep in enumerate(st.session_state.sales_reps):
                                if rep.id == rep_data['id']:
                                    # Update monthly settings if quota or incentive changed
                                    monthly_settings = rep.monthly_settings.copy()
                                    current_month = st.session_state.selected_month

                                    if new_quota != current_quota or new_incentive != current_incentive:
                                        if current_month not in monthly_settings:
                                            monthly_settings[current_month] = {}
                                        if new_quota != current_quota:
                                            monthly_settings[current_month]['quota'] = new_quota
                                        if new_incentive != current_incentive:
                                            monthly_settings[current_month]['target_incentive'] = new_incentive

                                    # Update sales data if rep ID changed
                                    old_rep_id = rep.id
                                    if new_rep_id != old_rep_id:
                                        for sale in st.session_state.sales_data:
                                            if sale.rep_id == old_rep_id:
                                                sale.rep_id = new_rep_id

                                    st.session_state.sales_reps[i] = SalesRep(
                                        id=new_rep_id,
                                        name=new_name.strip(),
                                        role_id=rep.role_id,
                                        manager_id=selected_manager_id,
                                        email=new_email.strip(),
                                        monthly_settings=monthly_settings
                                    )
                                    break

                            st.success(f"Updated {new_name}!")

                            # Recalculate commissions if needed
                            if st.session_state.sales_data:
                                st.info("Recalculating commissions due to rep changes...")
                                import time
                                time.sleep(0.5)

                            st.rerun()

            # Role summary
            role_totals = role_df.agg({
                'total_sales': 'sum',
                'total_commission': 'sum',
                'deals': 'sum',
                'attainment': 'mean'
            })

            st.write("**Role Summary:**")
            summary_cols = st.columns(4)
            with summary_cols[0]:
                st.metric("Total Sales", f"${role_totals['total_sales']:,.0f}")
            with summary_cols[1]:
                st.metric("Total Commission", f"${role_totals['total_commission']:,.2f}")
            with summary_cols[2]:
                st.metric("Total Deals", int(role_totals['deals']))
            with summary_cols[3]:
                st.metric("Avg Attainment", f"{role_totals['attainment']:.1f}%")

            st.divider()

        # Rep modal for create/edit
        if st.session_state.get('show_rep_modal', False):
            self.render_rep_modal()

    def render_rep_modal(self):
        """Render modal for creating/editing sales reps"""
        editing_rep = st.session_state.get('editing_rep')
        modal_title = "Edit Sales Representative" if editing_rep else "Create New Sales Representative"

        with st.form("rep_form"):
            st.subheader(modal_title)

            # Basic Information
            col1, col2 = st.columns(2)

            with col1:
                name = st.text_input(
                    "Full Name *",
                    value=editing_rep.name if editing_rep else "",
                    help="Enter the full name of the sales representative"
                )

                email = st.text_input(
                    "Email Address *",
                    value=editing_rep.email if editing_rep else "",
                    help="Enter a valid email address"
                )

            with col2:
                # Role selection
                role_options = [(role.id, role.name) for role in st.session_state.roles]
                role_names = [role[1] for role in role_options]

                current_role_idx = 0
                if editing_rep:
                    try:
                        current_role_idx = next(
                            i for i, (role_id, _) in enumerate(role_options) if role_id == editing_rep.role_id)
                    except StopIteration:
                        current_role_idx = 0

                selected_role_name = st.selectbox(
                    "Role *",
                    options=role_names,
                    index=current_role_idx,
                    help="Select the rep's role (determines commission structure)"
                )

                selected_role_id = next(
                    role_id for role_id, role_name in role_options if role_name == selected_role_name)

                # Manager selection
                manager_options = [("", "None")] + [(rep.id, rep.name) for rep in st.session_state.sales_reps if
                                                    not editing_rep or rep.id != editing_rep.id]
                manager_names = [name for _, name in manager_options]

                current_manager_idx = 0
                if editing_rep and editing_rep.manager_id:
                    try:
                        current_manager_idx = next(
                            i for i, (mgr_id, _) in enumerate(manager_options) if mgr_id == editing_rep.manager_id)
                    except StopIteration:
                        current_manager_idx = 0

                selected_manager_name = st.selectbox(
                    "Manager",
                    options=manager_names,
                    index=current_manager_idx,
                    help="Select the rep's manager (optional)"
                )

                selected_manager_id = next(
                    mgr_id for mgr_id, mgr_name in manager_options if mgr_name == selected_manager_name)
                selected_manager_id = selected_manager_id if selected_manager_id else None

            # Monthly Settings Override (Optional)
            st.divider()
            st.subheader("Monthly Settings Override (Optional)")
            st.caption("Leave blank to use role defaults")

            col1, col2 = st.columns(2)

            with col1:
                custom_quota = st.number_input(
                    "Custom Monthly Quota",
                    min_value=0.0,
                    value=0.0,
                    step=1000.0,
                    help="Override the role's default quota for this rep"
                )

            with col2:
                custom_incentive = st.number_input(
                    "Custom Target Incentive",
                    min_value=0.0,
                    value=0.0,
                    step=100.0,
                    help="Override the role's default target incentive for this rep"
                )

            # Form buttons
            st.divider()
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                if st.form_submit_button("💾 Save Rep", type="primary"):
                    # Validation
                    if not name.strip():
                        st.error("Name is required")
                        return

                    if not email.strip():
                        st.error("Email is required")
                        return

                    # Check for duplicate names (except when editing)
                    existing_names = [rep.name.lower() for rep in st.session_state.sales_reps if
                                      not editing_rep or rep.id != editing_rep.id]
                    if name.lower().strip() in existing_names:
                        st.error("A rep with this name already exists")
                        return

                    # Prepare monthly settings
                    monthly_settings = {}
                    if custom_quota > 0 or custom_incentive > 0:
                        current_month = st.session_state.selected_month
                        monthly_settings[current_month] = {}
                        if custom_quota > 0:
                            monthly_settings[current_month]['quota'] = custom_quota
                        if custom_incentive > 0:
                            monthly_settings[current_month]['target_incentive'] = custom_incentive

                    if editing_rep:
                        # Update existing rep
                        for i, rep in enumerate(st.session_state.sales_reps):
                            if rep.id == editing_rep.id:
                                # Merge monthly settings
                                existing_monthly = rep.monthly_settings.copy()
                                existing_monthly.update(monthly_settings)

                                st.session_state.sales_reps[i] = SalesRep(
                                    id=editing_rep.id,
                                    name=name.strip(),
                                    role_id=selected_role_id,
                                    manager_id=selected_manager_id,
                                    email=email.strip(),
                                    monthly_settings=existing_monthly
                                )
                                break

                        st.success(f"Updated rep '{name}'!")
                    else:
                        # Create new rep
                        new_rep = SalesRep(
                            id=str(uuid.uuid4()),
                            name=name.strip(),
                            role_id=selected_role_id,
                            manager_id=selected_manager_id,
                            email=email.strip(),
                            monthly_settings=monthly_settings
                        )
                        st.session_state.sales_reps.append(new_rep)
                        st.success(f"Created new rep '{name}'!")

                    # Close modal
                    st.session_state.show_rep_modal = False
                    if 'editing_rep' in st.session_state:
                        del st.session_state.editing_rep

                    st.rerun()

            with col2:
                if st.form_submit_button("🔍 Preview Changes"):
                    # Show preview of what will be saved
                    st.subheader("Preview:")
                    selected_role = next((role for role in st.session_state.roles if role.id == selected_role_id), None)

                    preview_data = {
                        "Name": name.strip(),
                        "Email": email.strip(),
                        "Role": selected_role.name if selected_role else "Unknown",
                        "Manager": selected_manager_name if selected_manager_name != "None" else "None",
                        "Default Quota": f"${selected_role.default_monthly_quota:,.0f}" if selected_role else "N/A",
                        "Default Incentive": f"${selected_role.default_target_incentive:,.0f}" if selected_role else "N/A"
                    }

                    if custom_quota > 0:
                        preview_data["Custom Quota"] = f"${custom_quota:,.0f}"
                    if custom_incentive > 0:
                        preview_data["Custom Incentive"] = f"${custom_incentive:,.0f}"

                    for key, value in preview_data.items():
                        st.write(f"**{key}:** {value}")

            with col3:
                if st.form_submit_button("❌ Cancel"):
                    st.session_state.show_rep_modal = False
                    if 'editing_rep' in st.session_state:
                        del st.session_state.editing_rep
                    st.rerun()

    def render_individual_view(self):
        """Render individual rep view tab - USES CENTRALIZED DATA"""
        st.subheader("Individual Rep Dashboard")

        rep_summary = self.get_rep_summary(st.session_state.selected_month)

        if not rep_summary:
            st.info("No sales rep data available.")
            return

        # Rep selector
        rep_names = [rep['name'] for rep in rep_summary]
        selected_rep_name = st.selectbox("Select a rep:", [''] + rep_names)

        if not selected_rep_name:
            st.info("Please select a sales representative to view their dashboard.")
            return

        selected_rep = next(rep for rep in rep_summary if rep['name'] == selected_rep_name)
        monthly_data = self.get_monthly_data(st.session_state.selected_month)
        rep_sales = [sale for sale in monthly_data if sale.rep_id == selected_rep['id']]

        # Rep header
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"## {selected_rep['name']}")
            st.markdown(f"**Role:** {selected_rep['role']}")
            st.markdown(f"**Email:** {selected_rep['email']}")
            st.markdown(f"**Month:** {st.session_state.selected_month}")

        with col2:
            if st.button("📊 Export to CSV"):
                if rep_sales:
                    df = pd.DataFrame([asdict(sale) for sale in rep_sales])
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{selected_rep['name']}_sales_{st.session_state.selected_month}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No sales data to export.")

        # Rep metrics from centralized data
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Monthly Sales", f"${selected_rep['total_sales']:,.0f}")
        with col2:
            st.metric("Commission", f"${selected_rep['total_commission']:,.0f}")
        with col3:
            attainment = (selected_rep['total_sales'] / selected_rep['quota'] * 100) if selected_rep['quota'] > 0 else 0
            st.metric("Quota Attainment", f"{attainment:.1f}%")
        with col4:
            st.metric("Deals Closed", selected_rep['deals'])

        # Show data connection status
        all_rep_sales = [sale for sale in st.session_state.sales_data if sale.rep_id == selected_rep['id']]
        if all_rep_sales:
            st.info(
                f"📊 Showing {len(rep_sales)} sales for {st.session_state.selected_month} out of {len(all_rep_sales)} total sales for this rep")

        # Sales details from centralized data
        st.subheader(f"Sales Details - {st.session_state.selected_month}")

        if rep_sales:
            sales_df = pd.DataFrame([{
                'Date': sale.date,
                'Product': sale.product,
                'Customer': sale.customer,
                'Sale Amount': f"${sale.sale_amount:,.0f}",
                'Commission': f"${sale.commission:,.2f}",
                'Commission Status': '✅ Calculated' if sale.commission > 0 else '⏳ Pending'
            } for sale in rep_sales])

            st.dataframe(sales_df, use_container_width=True)
        else:
            st.info(f"No sales data found for this rep in {st.session_state.selected_month}.")

        # Historical performance chart
        if all_rep_sales:
            st.subheader("Historical Performance")

            # Group sales by month
            monthly_performance = {}
            for sale in all_rep_sales:
                month = sale.date[:7]  # Extract YYYY-MM
                if month not in monthly_performance:
                    monthly_performance[month] = {'sales': 0, 'commission': 0, 'deals': 0}
                monthly_performance[month]['sales'] += sale.sale_amount
                monthly_performance[month]['commission'] += sale.commission
                monthly_performance[month]['deals'] += 1

            if len(monthly_performance) > 1:
                perf_df = pd.DataFrame([
                    {'Month': month, 'Sales': data['sales'], 'Commission': data['commission'], 'Deals': data['deals']}
                    for month, data in sorted(monthly_performance.items())
                ])

                col1, col2 = st.columns(2)

                with col1:
                    fig_sales = px.line(perf_df, x='Month', y='Sales', title='Monthly Sales Performance')
                    fig_sales.update_traces(line=dict(color='#1f77b4', width=3))
                    st.plotly_chart(fig_sales, use_container_width=True)

                with col2:
                    fig_commission = px.line(perf_df, x='Month', y='Commission', title='Monthly Commission Performance')
                    fig_commission.update_traces(line=dict(color='#ff7f0e', width=3))
                    st.plotly_chart(fig_commission, use_container_width=True)

    def run(self):
        """Main application runner with CROSS-TAB COMMUNICATION"""
        self.render_header()

        # Create tabs with data connectivity indicators
        tab_names = [
            "📊 Dashboard",
            "📤 Upload Data",
            "📋 Data Summary",
            "🎭 Roles & Rules",
            "👥 Sales Reps",
            "📈 Individual View"
        ]

        tabs = st.tabs(tab_names)

        # Add global data status in sidebar
        with st.sidebar:
            st.header("Data Status")

            if st.session_state.sales_data:
                st.success(f"✅ {len(st.session_state.sales_data)} sales records loaded")

                calculated_count = len([sale for sale in st.session_state.sales_data if sale.commission > 0])
                if calculated_count == len(st.session_state.sales_data):
                    st.success("✅ All commissions calculated")
                elif calculated_count > 0:
                    st.warning(f"⚠️ {calculated_count}/{len(st.session_state.sales_data)} commissions calculated")
                else:
                    st.error("❌ No commissions calculated")

                if st.session_state.commission_last_calculated:
                    calc_time = st.session_state.commission_last_calculated.strftime("%H:%M:%S")
                    st.caption(f"Last calculated: {calc_time}")

                # Quick actions in sidebar
                st.header("Quick Actions")

                if st.button("🧮 Recalculate All", key="sidebar_recalc"):
                    with st.spinner("Recalculating..."):
                        self.calculate_commissions()
                        st.success("Recalculated!")
                        st.rerun()

                if st.button("📊 Go to Upload Tab", key="sidebar_upload"):
                    st.session_state.active_tab = "upload"
                    st.rerun()
            else:
                st.error("❌ No sales data loaded")
                st.info("Use the Upload Data tab to get started")

        # Render individual tabs
        with tabs[0]:  # Dashboard
            self.render_dashboard()

        with tabs[1]:  # Upload Data - PRIMARY DATA SOURCE
            self.render_upload_data()

        with tabs[2]:  # Roles & Rules
            self.render_data_summary()

        with tabs[3]:  # Roles & Rules
            self.render_roles()

        with tabs[4]:  # Sales Reps
            self.render_sales_reps()

        with tabs[5]:  # Individual View
            self.render_individual_view()


# Run the application
if __name__ == "__main__":
    app = CommissionApp()
    app.run()