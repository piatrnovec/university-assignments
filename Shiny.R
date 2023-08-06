library(shiny)
library(shinythemes)
library(ggplot2)
library(caret)
library(pROC)
library(ROCR)
library(reshape2)
library(lmtest)
library(MASS)

df <- read.csv("LOGIT.csv", sep = ";")

# TRATAMIENTO DE DATOS----------------------------------------------------------
# Eliminar columnas con valores iguales o muchos valores nulos
variables_eliminadas <- c("v1", "v6", "v10", "v12", "v14", "v15", "v24", "v25", "v29", "v30", "v31", "v33", "v34")
df <- df[, !(names(df) %in% variables_eliminadas)]

# Reemplazar valores texto a numerico
df$v20 <- ifelse(df$v20 == "S", 1, 0)

# Reemplazar valores faltantes con 0
df[df == ""] <- NA
df[is.na(df)] <- 0

# Obtener las columnas desde la tercera columna hasta la última
columnas <- names(df)[5:ncol(df)-1]

# Bucle para reemplazar ',' por '.'
for (col in columnas) {
  df[[col]] <- gsub(",", ".", df[[col]])
}

# Bucle para convertir las columnas a formato numérico
for (col in columnas) {
  df[[col]] <- as.numeric(df[[col]])
}

# Preprocesamiento de datos
variables_continuas <- names(df)[5:(ncol(df) - 1)]
df_variables <- df[, variables_continuas]

ui <- shinyUI(
  fluidPage(
    tags$head(
      tags$style(
        HTML("
          body {
            background-color: pink;
          }
          .title {
            font-size: 24px;
            font-weight: bold;
            text-decoration: underline;
            text-align: center;
            margin-bottom: 20px;
          }
          .plot-container {
            border: 1px solid gray;
            padding: 10px;
            border-radius: 5px;
          }
          
        ")
      )
    ),
    titlePanel(tags$h2(class = "title", "Mini Desarrollo Modelo Logit con Shiny")),
    fluidRow(
      column(
        12,
        align = "center",
        h2("Bienvenidos a nuestra presentación de Mini Desarrollo Modelo Logit con Shiny!")
      )
    ),
    fluidRow(
      column(
        12,
        align = "center",
        actionButton("generate_data_btn", "Lee Datos")
      )
    ),
    fluidRow(
      column(
        2,
        selectInput(
          "dep_var",
          "Dependent Variable",
          choices = "target",
          selected = "target"
        )
      ),
      column(
        2,
        selectInput(
          "ind_vars",
          "Independent Variables",
          choices = setdiff(columnas, "target"),
          multiple = TRUE,
          selected = "v2"
        )
      ),
      column(
        2,
        selectInput(
          "lr_var",
          "LR-Test (Select Vars to Exclude)",
          choices = setdiff(columnas, "target"),
          multiple = TRUE,
          selected = "v2"
        )
      ),
      column(
        2,
        selectInput(
          "plot_var",
          "Plotting Variable",
          choices = columnas,
          selected = "v2"
        )
      ),
      column(
        2,
        selectInput(
          "facet_var",
          "Facet Variable",
          choices = c("target", columnas),
          selected = "target"
        )
      )
    ),
    fluidRow(
      column(
        2,
        selectInput(
          "var_selection_method",
          "Variable Selection Method",
          choices = c("Forward", "Stepwise", "Backward"),
          selected = "Forward"
        )
      )
    ),
    fluidRow(
      column(
        12,
        div(class = "plot-container",
            plotOutput("plot1", height = "400px")
        )
      )
    ),
    fluidRow(
      column(
        6,
        verbatimTextOutput("results")
      ),
      column(
        6,
        verbatimTextOutput("lr_test")
      )
    ),
    fluidRow(
      column(
        12,
        div(class = "plot-container",
            plotOutput("correlation", height = "400px")
        )
      )
    ),
    fluidRow(
      column(
        12,
        div(class = "plot-container",
            plotOutput("roc", height = "400px")
        )
      )
    )
  )
)

server <- function(input, output, session) {
  data_loaded <- reactiveVal(FALSE)
  df_train <- reactiveVal()
  df_test <- reactiveVal()
  var_selection_method <- reactiveVal("Forward")
  
  observeEvent(input$generate_data_btn, {
    data_loaded(TRUE)
    
    # Split data into training and testing sets
    prop_train <- 0.7  # Proportion of data for training
    set.seed(123)  # Set seed for reproducibility
    train_indices <- createDataPartition(df$target, p = prop_train, list = FALSE)
    df_train(df[train_indices, ])
    df_test(df[-train_indices, ])
  })
  
  # Reactive value for storing the model
  model <- reactive({
    if (data_loaded() && !is.null(input$ind_vars) && length(input$ind_vars) > 0) {
      dep_var <- input$dep_var
      ind_vars <- input$ind_vars
      
      if (var_selection_method() == "Forward") {
        form <- as.formula(paste(dep_var, "~", paste(ind_vars, collapse = "+")))
        glm(form, family = binomial(link = "logit"), data = df_train())
      } else if (var_selection_method() == "Stepwise") {
        form <- as.formula(paste(dep_var, "~", paste(ind_vars, collapse = "+")))
        stepAIC(glm(form, family = binomial(link = "logit"), data = df_train()), direction = "both", trace = FALSE)
      } else if (var_selection_method() == "Backward") {
        form <- as.formula(paste(dep_var, "~", paste(ind_vars, collapse = "+")))
        stepAIC(glm(form, family = binomial(link = "logit"), data = df_train()), direction = "backward", trace = FALSE)
      }
    }
  })
  
  output$results <- renderPrint({
    model_summary <- summary(model())
    if (!is.null(model_summary))
      model_summary
  })
  
  output$lr_test <- renderPrint({
    if (data_loaded() && !is.null(input$ind_vars) && length(input$ind_vars) > 0) {
      dep_var <- input$dep_var
      ind_vars <- input$ind_vars
      lr_var <- input$lr_var
      if (is.null(lr_var)) {
        lr_var <- character(0)
      }
      lr_var <- setdiff(ind_vars, lr_var)
      if (length(lr_var) > 0) {
        form <- as.formula(paste(dep_var, "~", paste(lr_var, collapse = "+")))
        model_reduced <- glm(form, family = binomial(link = "logit"), data = df_train())
        lrtest(model_reduced, model())
      }
    }
  })
  
  output$plot1 <- renderPlot({
    suppressWarnings({
      if (data_loaded() && !is.null(input$plot_var) && !is.null(input$facet_var)) {
        plot_var <- input$plot_var
        facet_var <- input$facet_var
        ggplot(df_train(), aes(x = .data[[plot_var]])) +
          geom_density(aes(color = target), alpha = 0.6) +
          facet_wrap(~ .data[[facet_var]]) +
          theme_bw() +
          theme(legend.position = "top")
      }
    })
  })
  
  output$correlation <- renderPlot({
    if (data_loaded() && !is.null(input$ind_vars) && length(input$ind_vars) > 0) {
      ind_vars <- input$ind_vars
      corr_matrix <- cor(df_train()[, ind_vars])
      ggplot(melt(corr_matrix), aes(x = Var2, y = Var1, fill = value)) +
        geom_tile() +
        scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                             midpoint = 0, limit = c(-1, 1), space = "Lab",
                             name = "Correlation") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
    }
  })
  
  output$roc <- renderPlot({
    if (data_loaded()) {
      # ROC Curve
      model_roc <- roc(df_test()$target, predict(model(), newdata = df_test(), type = "response"))
      
      auc_value <- model_roc$auc
      
      if (auc_value > 0.015) {
        auc_label <- paste0("AUC: ", round(auc_value, 3))
      } else {
        auc_label <- paste0("AUC: ", round(auc_value, 3))
      }
      
      plot(model_roc, print.thres = "best", print.thres.best.method = "closest.topleft", print.auc = FALSE)
      text(0.5, 0.9, auc_label, font = ifelse(auc_value > 0.015, 2, 1), cex = 1.2, adj = 0.5)
    }
  })
  observeEvent(input$var_selection_method, {
    var_selection_method(input$var_selection_method)
  })
}

shinyApp(ui, server)