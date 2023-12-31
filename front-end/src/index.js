import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';

import { BrowserRouter, Routes, Route } from "react-router-dom";
import LandingPage from "./pages/landing_page";
import ResultsPage from "./pages/results_page";
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
    <BrowserRouter>
        <Routes>
            <Route path="/" exact element={<LandingPage />}></Route>
            <Route path="/results" exact element={<ResultsPage />}></Route>
            {/*<Route path="/help" exact element={<Help />}></Route>*/}
            {/*<Route path="/about" exact element={<About />}></Route>*/}
            {/*<Route path="/results" exact element={<Result />}></Route>*/}
            {/*<Route path="*" element={<PageNotFound />}></Route>*/}
        </Routes>
    </BrowserRouter>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
